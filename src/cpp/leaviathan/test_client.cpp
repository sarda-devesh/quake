#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"

#include <communication/compute_client.h>
#include <quake_index.h>

#include <arrow/api.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <filesystem>

ABSL_FLAG(std::string, compute_address, "localhost:9001", "Address of the compute node we want to test");

void benchmark_sift1m() { 
    // Create the communication client
    std::string compute_address = absl::GetFlag(FLAGS_compute_address);
    std::shared_ptr<ComputeClient> compute_client = std::make_shared<ComputeClient>(compute_address);

    // Now run a query against the index
    int num_test_queries = 1;
    int dimension = 128;
    int nprobe = 20;
    int k = 10;
    torch::Tensor search_queries = torch::randn({num_test_queries, dimension}, torch::kFloat32);
    std::string sift_index_dir("/working_dir/quake_workloads/vary_storage_nodes_sift1m_read_only/init_indexes");

    // Load the index locally
    int local_loaded_index_id = compute_client->load_existing_index(sift_index_dir, true, 16, true);
    std::cout << "Locally loaded index has id of " << local_loaded_index_id << std::endl;

    std::shared_ptr<SearchIndexResult> loaded_local_query_result = compute_client->search_index(local_loaded_index_id, search_queries, k, nprobe);
    if(loaded_local_query_result->query_sucessful) { 
        std::cout << "LOCAL SUCESSS: Got result in " << loaded_local_query_result->total_query_time_ns << " ns" << std::endl;
    } else {
        std::cout << "LOCAL FAILURE: Error Message - " << loaded_local_query_result->error_message << std::endl;
    }

    /*
    // Load the index globally
    int global_loaded_index_id = compute_client->load_existing_index(sift_index_dir, false);
    std::cout << "Globally loaded index has id of " << global_loaded_index_id << std::endl;

    std::shared_ptr<SearchIndexResult> loaded_global_query_result = compute_client->search_index(global_loaded_index_id, search_queries, k, nprobe);
    if(loaded_global_query_result->query_sucessful) { 
        std::cout << "GLOBAL SUCESSS: Got result in " << loaded_global_query_result->total_query_time_ns << " ns" << std::endl;
    } else {
        std::cout << "GLOBAL FAILURE: Error Message - " << loaded_global_query_result->error_message << std::endl;
    }
    */
}

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    benchmark_sift1m();

    /*
    // Create the communication client
    std::string compute_address = absl::GetFlag(FLAGS_compute_address);
    std::shared_ptr<ComputeClient> compute_client = std::make_shared<ComputeClient>(compute_address);

    // Initialize the test parameters
    int d = 128;
    int num_vec = 20;
    int num_partitions = 2;
    int num_test_queries = 2; int results_per_query = 2;
    torch::Tensor search_queries = torch::randn({num_test_queries, d}, torch::kFloat32);

    int global_index_id = compute_client->create_new_index(d, num_vec, num_partitions);
    std::cout << "Got global index with id of " << global_index_id << std::endl;

    // Run some queries against it and log the result
    std::shared_ptr<SearchIndexResult> global_query_result = compute_client->search_index(global_index_id, search_queries, results_per_query, 1);
    if(global_query_result->query_sucessful) { 
        std::cout << "NEW GLOBAL SUCESSS: Ids - " << global_query_result->ids << ", Distances - " << global_query_result->distances << std::endl;
    } else {
        std::cout << "NEW GLOBAL FAILURE: Error Message - " << global_query_result->error_message << std::endl;
    }

    // Now do the same with a local test index
    int local_index_id = compute_client->create_new_index(d, num_vec, num_partitions, true);
    std::cout << "Got local index id of " << local_index_id << std::endl;

    std::shared_ptr<SearchIndexResult> local_query_result = compute_client->search_index(local_index_id, search_queries, results_per_query, 1);
    if(local_query_result->query_sucessful) { 
        std::cout << "NEW LOCAL SUCESSS: Ids - " << local_query_result->ids << ", Distances - " << local_query_result->distances << std::endl;
    } else {
        std::cout << "NEW LOCAL FAILURE: Error Message - " << local_query_result->error_message << std::endl;
    }

    // Build an index and then save it to a local directory
    std::filesystem::path test_index_save_dir = "test_demo_index";
    if(std::filesystem::exists(test_index_save_dir)) { 
        std::filesystem::remove_all(test_index_save_dir);
    }

    std::shared_ptr<QuakeIndex> test_index = std::make_shared<QuakeIndex>();
    torch::Tensor build_vectors = torch::randn({num_vec, d}, torch::kFloat32);
    torch::Tensor build_ids = torch::arange(0, num_vec, torch::kInt64);
    std::shared_ptr<IndexBuildParams> build_params = std::make_shared<IndexBuildParams>();
    build_params->dimension = d;
    build_params->nlist = num_partitions;
    test_index->build(build_vectors, build_ids, build_params);
    
    std::string test_index_path = test_index_save_dir.string();
    test_index->save(test_index_path);
    std::cout << "Saved test existing index to " << test_index_save_dir << std::endl;

    // Now try to load that index locally on the compute node and run queries against it
    int local_loaded_index_id = compute_client->load_existing_index(test_index_path, true);
    std::cout << "Locally loaded index has id of " << local_loaded_index_id << std::endl;

    std::shared_ptr<SearchIndexResult> loaded_local_query_result = compute_client->search_index(local_loaded_index_id, search_queries, results_per_query, 1);
    if(loaded_local_query_result->query_sucessful) { 
        std::cout << "LOADED LOCAL SUCESSS: Ids - " << loaded_local_query_result->ids << ", Distances - " << loaded_local_query_result->distances << std::endl;
    } else {
        std::cout << "LOADED LOCAL FAILURE: Error Message - " << loaded_local_query_result->error_message << std::endl;
    }

    // Now try to load the same index but try to distribute it globally
    int global_loaded_index_id = compute_client->load_existing_index(test_index_path);
    std::cout << "Globally loaded index has id of " << global_loaded_index_id << std::endl;

    std::shared_ptr<SearchIndexResult> loaded_global_query_result = compute_client->search_index(global_loaded_index_id, search_queries, results_per_query, 1);
    if(loaded_global_query_result->query_sucessful) { 
        std::cout << "LOADED GLOBAL SUCESSS: Ids - " << loaded_global_query_result->ids << ", Distances - " << loaded_global_query_result->distances << std::endl;
    } else {
        std::cout << "LOADED GLOBAL FAILURE: Error Message - " << loaded_global_query_result->error_message << std::endl;
    }
    */
}