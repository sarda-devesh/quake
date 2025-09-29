//
// Created by Devesh on 09/22/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef COMPUTE_CLIENT_H
#define COMPUTE_CLIENT_H

#include <grpcpp/grpcpp.h>
#include <iostream>
#include <memory>
#include <string>
#include <torch/torch.h>

#include "compute_node.grpc.pb.h" 

using computenode::ComputeNode;

struct SearchIndexResult {
    bool query_sucessful; // Boolean representing if the query was sucessful or not
    std::string error_message; // If the query failed, the error message returned by the compute node
    torch::Tensor ids; // If the query successed, the resulting ids (torch::kInt64) for the search query. Shape: [num_queries, k]
    torch::Tensor distances; // If the query successed, the resulting distances (torch::kFloat32) for the search query. Shape: [num_queries, k]

    SearchIndexResult() = default;
};

/**
 * @brief Wrapper to interact with a compute node via gRPC.
 *
 * The ComputeClient is a client-side wrapper that facilitates communication
 * with a compute node via gRPC. All communication with a compute node
 * should be done through this class.
 */
class ComputeClient { 
public: 
    /** 
    * @brief Constructor for ComputeClient
    * @param compute_address The address of the compute node we want to communicate with (e.g., "localhost:5051").
    */
    ComputeClient(std::string compute_address);

    /**
    * @brief Method to create a new index in this compute node
    * 
    * @param d The dimension of the vectors in the new index
    * @param num_vectors The number of vectors that should be created in the new index
    * @param num_partitions The number of partitions to initialize the new vector with
    * 
    * @return The id associated with the newly created index 
    */
   int create_new_index(int d, int num_vectors, int num_partitions);

   /**
    * @brief Method to search an existing index in the compute node
    * 
    * @param index_id The id of the index we are performing the search on
    * @param search_vectors Tensor representing the vectors we are searching for in the index. Shape: [num_queries, dimension]
    * @param k The number of results to return per vector (Deafults to 1)
    * @param nprobe The number of partitions to check (Defaults to 1)
    * @param recall_target The recall target for this search query (Defaults to -1.0, meaning no adpative search)
    * 
    * @return An instance of SearchIndexResult representing the query result
    */
   std::shared_ptr<SearchIndexResult> search_index(int index_id, torch::Tensor search_vectors, int k = 1, int nprobe = 1, float recall_target = -1.0);

private:
    std::unique_ptr<ComputeNode::Stub> stub_; ///< gRPC stub for making calls to the compute node
};

#endif // COMPUTE_CLIENT_H