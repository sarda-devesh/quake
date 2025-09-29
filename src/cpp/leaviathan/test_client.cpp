#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"

#include <communication/compute_client.h>

#include <arrow/api.h>
#include <iostream>
#include <cstdlib>
#include <cstring>

ABSL_FLAG(std::string, compute_address, "localhost:9001", "Address of the compute node we want to test");


int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);

    // Create the communication client
    std::string compute_address = absl::GetFlag(FLAGS_compute_address);
    std::shared_ptr<ComputeClient> compute_client = std::make_shared<ComputeClient>(compute_address);

    // Create a new test index
    int d = 128;
    int num_vec = 20;
    int num_partitions = 2;
    int index_id = compute_client->create_new_index(d, num_vec, num_partitions);
    std::cout << "Got index with id of " << index_id << std::endl;

    // Run some queries against it and log the result
    int num_test_queries = 2; int results_per_query = 2;
    torch::Tensor search_queries = torch::randn({num_test_queries, d}, torch::kFloat32);
    std::shared_ptr<SearchIndexResult> query_result = compute_client->search_index(index_id, search_queries, results_per_query, 1);
    if(query_result->query_sucessful) { 
        std::cout << "SUCESSS: Ids - " << query_result->ids << ", Distances - " << query_result->distances << std::endl;
    } else {
        std::cout << "FAILURE: Error Message - " << query_result->error_message << std::endl;
    }
}