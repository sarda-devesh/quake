import subprocess
from pathlib import Path
import torch
import pandas as pd
import yaml
import time
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from quake.utils import *
from quake.leaviathan_cluster import *
from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.quake import QuakeWrapper
from quake.index_wrappers.remote_index import RemoteIndexWrapper
from quake.workload_generator import DynamicWorkloadGenerator, WorkloadEvaluator

def build_index(config, workload_dir, index_save_dir):
    # Load dataset using the configuration.
    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path", "data")
    vectors, queries, gt = load_dataset(dataset_name, dataset_path)
    print("Building and saving index from dataset", dataset_name, "to", index_save_dir)

    # Initialize the dynamic workload generator.
    workload_cfg = config["workload"]
    base_experiment_name = config["name"]
    base_index_details = config["base_index"]
    metric = base_index_details["metric"]
    workload_gen = DynamicWorkloadGenerator(
        workload_dir=workload_dir,
        base_vectors=vectors,
        metric = metric,
        insert_ratio=workload_cfg["insert_ratio"],
        delete_ratio=workload_cfg["delete_ratio"],
        query_ratio=workload_cfg["query_ratio"],
        update_batch_size=workload_cfg["update_batch_size"],
        query_batch_size=workload_cfg["query_batch_size"],
        number_of_operations=workload_cfg["number_of_operations"],
        initial_size=workload_cfg["initial_size"],
        cluster_size=workload_cfg["cluster_size"],
        cluster_sample_distribution=workload_cfg["cluster_sample_distribution"],
        queries=queries,
        seed=config.get("seed", 1738),
    )

    # Generate the workload if it doesn't already exist.
    if not workload_gen.workload_exists():
        print("Generating workload...")
        workload_gen.generate_workload()
    else:
        print("Workload already exists; reusing generated workload.")

    # Load the dataset to build the index
    indices_path = workload_dir / "initial_indices.pt"
    vectors_path = workload_dir / "base_vectors.pt"
    build_vectors = torch.load(vectors_path, weights_only=True).to(torch.float32)
    build_indices = torch.load(indices_path, weights_only=True).to(torch.int64)
    build_vectors = build_vectors[build_indices]
    print("Loaded build vectors of size", build_vectors.size(), "and indicies of size", build_indices.size())

    # Build and save the index
    index = QuakeWrapper()
    num_partitions = base_index_details.get("nc", 1024)
    build_params = {
        "nc": num_partitions,
        "metric": metric,
        "num_workers": base_index_details["num_build_workers"],
    }
    index.build(build_vectors, ids=build_indices, **build_params)
    index.save(index_save_dir)
    print("Saved index with", num_partitions, "partitions to", index_save_dir)

def run_experiment_for_config(curr_config, global_variables, complete_config):
    is_heartbeat_experiment = curr_config["name"] == "baseline_rpc_latency"
    print("\nRunning experiment for config", curr_config)

    # Startup the cluster and give it time to bootup
    cluster = LeviathanClusterManager(complete_config["quake_build_dir"], curr_config["num_compute"], curr_config["num_storage"])
    assert cluster.start_cluster(), f"Failed to launch cluster for config {curr_config}"
    time.sleep(2)

    sucess = True
    try:
        # Load up the index on one of the compute nodes
        store_index_locally = str(curr_config.get("load_index_locally", "False")).lower() == "true"
        remote_index = RemoteIndexWrapper(cluster.get_compute_node_address()[0])
        if not is_heartbeat_experiment:
            store_index_on_disk = str(curr_config.get("store_index_on_disk", "False")).lower() == "true"
            if "num_search_workers" in curr_config:
                remote_index.load_index(global_variables["index_dir"], store_index_locally=store_index_locally, num_search_workers=curr_config["num_search_workers"], store_index_on_disk=store_index_on_disk)
            else:
                remote_index.load_index(global_variables["index_dir"], store_index_locally=store_index_locally, store_index_on_disk=store_index_on_disk)

        # Load the parameters needed to run the workload
        workload_dir = global_variables["workload_dir"]
        query_vectors = torch.load(workload_dir / "query_vectors.pt", weights_only=True)
        search_params = complete_config["search_params"]
        runbook = json.load(open(workload_dir / "runbook.json", "r"))
        operations_dir = workload_dir / "operations"

        # Now actually run the queries against the index
        k = search_params["k"]
        query_results = []
        operations = runbook["operations"]
        for operation_id in operations:
            # Load the operation queries and expected results
            if is_heartbeat_experiment:
                for i in range(operations[operation_id]["sample_size"]):
                    heartbeat_latency = remote_index.compute_client_.heartbeat()
                    curr_query_result = { 
                        "operation_id" : operation_id,
                        "local_query_id" : i,
                        "global_query_id" : len(query_results),
                        "latency_ns" : heartbeat_latency
                    }
                    query_results.append(curr_query_result)
                continue

            operation_id = int(operation_id)
            operation_ids = torch.load(operations_dir / f"{operation_id}.pt", weights_only=True)
            operation_queries = query_vectors[operation_ids]
            gt_ids = torch.load(operations_dir / f"{operation_id}_gt_ids.pt", weights_only=True)

            operation_query_counter = 0
            for query, expected_ids in zip(operation_queries, gt_ids):
                # Run each query indvidually
                expected_ids = expected_ids.reshape(1, -1)
                search_result = remote_index.search(query, **search_params)
                if not search_result.query_successful:
                    raise Exception("Query failed due to error " + str(search_result.error_message))
                
                # Record the recall and latency
                result_ids = search_result.ids.reshape(1, -1)
                query_recall = compute_recall(result_ids, expected_ids, k)
                query_latency_ns = search_result.total_query_time_ns

                curr_query_result = {
                    "operation_id" : operation_id,
                    "local_query_id" : operation_query_counter,
                    "global_query_id" : len(query_results),
                    "recall" : query_recall.item(),
                    "latency_ns" : query_latency_ns
                }
                query_results.append(curr_query_result)
                operation_query_counter += 1
    except Exception as e:
        print("[ERROR] Failed to run experiment with config", curr_config, " due to error", e)
        sucess = False
    finally:
        cluster.stop_cluster()
    
    # Exit if the experiment failed
    if not sucess:
        sys.exit(1)
    
    # If it sucessed then return the df to store
    return pd.DataFrame(query_results)

CONFIG_LABELS = {
    "baseline_rpc_latency" : "Boolean Echo RPC",
    "leviathan_1_1_local" : "Partitions in memory on Compute Node", 
    "leviathan_1_1_local_on_disk" : "Partitions in disk on Compute Node",
    "leviathan_1_1" : "Partitions in disk on 1 Storage Node", 
    "leviathan_1_2" : "Partitions in disk on 2 Storage Nodes",
    "leviathan_1_4" : "Partitions in disk on 4 Storage Nodes",
    "leviathan_1_8" : "Partitions in disk on 8 Storage Nodes",
}
NS_TO_MS = 1.0e6
def visualize_result_latency(experimental_results, save_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    for config_name, config_result in experimental_results.items():
        latencies = config_result["latency_ns"].values/NS_TO_MS
        sorted_latencies = np.sort(latencies)
        y_values = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        ax.plot(sorted_latencies, y_values, label=CONFIG_LABELS[config_name], linewidth=2)

    # Configure the axis details
    ax.set_xscale('log')
    ax.set_ylim(0.0, 1)
    ax.set_xlim(left=0)

    ax.set_title('Latency CDF on SIFT 1M (nc = 1024, nprobe = 20, k = 10)', fontsize=16)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('End to End Query Latency Latency (ms)', fontsize=14)
    ax.set_ylabel('Cumulative Probability', fontsize=14)
    ax.legend(fontsize=12)

    # Save the result
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print("Saved the latency graph to", save_path)

def run_experiment():
    # Load the overall configuration.
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / Path("configs/sift1m_read_only.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # If the index that we are going to load doesn't exist then create it
    base_experiment_name = config["name"]
    workload_dir = Path(config.get("workload_dir", "workloads")) / base_experiment_name
    index_dir = workload_dir / "init_indexes"

    if not index_dir.exists():
        # Build and save the index in the determined path
        build_index(config, workload_dir, index_dir)
    
    global_variables = {"index_dir" : index_dir, "workload_dir" : workload_dir}

    # Now run the experiment for different config
    overwrite = config.get("overwrite", False)
    experiment_result_dir = Path(config.get("results_dir", "results")) / base_experiment_name
    experiments_results = {}
    for curr_config in config.get("configs", []):
        config_name = curr_config["name"]
        results_dir = experiment_result_dir / config_name
        results_dir.mkdir(parents=True, exist_ok=True)
        result_file = results_dir / "results.csv"

        if result_file.exists() and not overwrite:
            print(f"Results already exist for config '{config_name}'")
            df = pd.read_csv(result_file)
            experiments_results[config_name] = df
        else:
            results_df = run_experiment_for_config(curr_config, global_variables, config)
            results_df.to_csv(result_file, index=False)
            experiments_results[config_name] = results_df
            print("Saved result for config", config_name, "to", result_file)
    
    # Now graph the result
    visualize_result_latency(experiments_results, experiment_result_dir / "config_latencies.png")

if __name__ == "__main__":
    run_experiment()

"""
configs:
  - name: baseline_rpc_latency
    load_index_locally: "True"
    num_compute: 1
    num_storage: 0
    num_search_workers: 16

  - name: leviathan_1_1_local
    load_index_locally: "True"
    num_compute: 1
    num_storage: 0
    num_search_workers: 16
 
  - name: leviathan_1_1_local_on_disk
    load_index_locally: "True"
    store_index_on_disk: "True"
    num_compute: 1
    num_storage: 0
    num_search_workers: 16

  - name: leviathan_1_1
    load_index_locally: "False"
    num_compute: 1
    num_storage: 1
    num_search_workers: 16
  
  - name: leviathan_1_2
    load_index_locally: "False"
    num_compute: 1
    num_storage: 2
    num_search_workers: 16
  
  - name: leviathan_1_4
    load_index_locally: "False"
    num_compute: 1
    num_storage: 4
    num_search_workers: 16
  
  - name: leviathan_1_8
    load_index_locally: "False"
    num_compute: 1
    num_storage: 8
    num_search_workers: 16
"""