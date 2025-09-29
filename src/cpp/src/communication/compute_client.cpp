//
// Created by Devesh on 09/22/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include "communication/compute_client.h"

#include <cassert>

using grpc::Status;
using computenode::NewIndexRequest;
using computenode::NewIndexReply;
using computenode::SearchIndexRequest;
using computenode::SearchIndexReply;

ComputeClient::ComputeClient(std::string compute_address) {
    // Create a gRPC channel to the compute node
    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(compute_address, grpc::InsecureChannelCredentials());
    stub_ = ComputeNode::NewStub(channel);
}

int ComputeClient::create_new_index(int d, int num_vectors, int num_partitions) { 
    grpc::ClientContext context;

    // Set the request parameters
    NewIndexRequest create_request;
    create_request.set_vector_dimension(d);
    create_request.set_num_vectors(num_vectors);
    create_request.set_num_clusters(num_partitions); 

    // Make the request
    NewIndexReply create_response; 
    grpc::Status status = stub_->CreateNewIndex(&context, create_request, &create_response);
    assert(status.ok());

    return create_response.index_id();
}

std::shared_ptr<SearchIndexResult> ComputeClient::search_index(int index_id, torch::Tensor search_vectors, int k, int nprobe, float recall_target) { 
    // Create the request
    grpc::ClientContext context;
    SearchIndexRequest search_request;
    search_request.set_index_id(index_id);
    search_request.set_top_k(k);
    search_request.set_num_probe(nprobe);
    search_request.set_recall_target(recall_target);

    torch::Tensor flattened_vectors = torch::flatten(search_vectors);
    int num_elements = flattened_vectors.numel();
    float* vectors_ptr = flattened_vectors.data_ptr<float>();
    auto* search_request_vectors = search_request.mutable_search_vectors();
    search_request_vectors->Clear(); search_request_vectors->Reserve(num_elements);
    search_request_vectors->Add(vectors_ptr, vectors_ptr + num_elements);

    // Make the request and construct the SearchIndexResult based on the respose
    SearchIndexReply search_response;
    grpc::Status status = stub_->SearchIndex(&context, search_request, &search_response);

    std::shared_ptr<SearchIndexResult> result = std::make_shared<SearchIndexResult>();
    result->query_sucessful = status.ok();
    if(result->query_sucessful) { 
        // Convert the results to tensors
        auto result_ids_field = search_response.result_ids();
        int num_search_vectors = result_ids_field.size()/k;
        result->ids = torch::from_blob(const_cast<int64_t*>(result_ids_field.data()), {num_search_vectors, k}, torch::kInt64).clone();

        auto result_distances_field = search_response.result_distances();
        result->distances = torch::from_blob(const_cast<float*>(result_distances_field.data()), {num_search_vectors, k}, torch::kFloat32).clone();
    } else { 
        // Record the error messsage
        result->error_message = status.error_message();
    }

    return result;
}