#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"

#include <communication/compute_client.h>
#include <quake_index.h>
#include <common.h>

#include <arrow/api.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <filesystem>

ABSL_FLAG(std::string, compute_address, "localhost:9001", "Address of the compute node we want to test");
ABSL_FLAG(std::string, index_type, "local_disk", "The type of index we want to use for the experiment");

constexpr uint32_t random_number_seed = 1379;

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    torch::manual_seed(random_number_seed);

    // Create the communication client
    std::string compute_address = absl::GetFlag(FLAGS_compute_address);
    std::shared_ptr<ComputeClient> compute_client = std::make_shared<ComputeClient>(compute_address);

    // Set the experiment parameters
    int num_test_queries = 100;
    int dimension = 128;
    int nprobe = 20;
    int k = 10;
    std::string sift_index_dir("/working_dir/quake_workloads/vary_storage_nodes_sift1m_read_only/init_indexes");

    // Get the index 
    int index_id;
    std::string index_type = absl::GetFlag(FLAGS_index_type);
    if(index_type == "local_disk") { 
        index_id = compute_client->load_existing_index(sift_index_dir, true, 1, true);
    } else if(index_type == "local_inmem") { 
        index_id = compute_client->load_existing_index(sift_index_dir, true, 1, false);
    } else if(index_type == "remote_inmem") { 
        index_id = compute_client->load_existing_index(sift_index_dir, false, 1, false);
    } else { 
        std::cerr << "Invalid index type of " << index_type << std::endl;
        return -1;
    }
    std::cout << "Got an index id of " << index_id << std::endl;

    // Now run queries against the index
    MetricStore latency_metric_values("client_latency_ms");
    for(int i = 0; i < num_test_queries; i++) { 
        torch::Tensor search_query = torch::randn({1, dimension}, torch::kFloat32);
        std::shared_ptr<SearchIndexResult> query_result = compute_client->search_index(index_id, search_query, k, nprobe);
        if(!query_result->query_sucessful) { 
            std::cerr << "Query failed due to error " << query_result->error_message << std::endl;
            exit(1);
        }
        latency_metric_values.add_value(query_result->total_query_time_ns/MS_TO_NS);
    }

    // Print local as well as system metrics
    latency_metric_values.print_metric();
    compute_client->print_metrics();

    return 0;
}