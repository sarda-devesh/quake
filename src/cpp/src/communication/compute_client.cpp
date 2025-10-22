//
// Created by Devesh on 09/22/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include "communication/compute_client.h"

#include <cassert>
#include <filesystem>
#include <chrono>

using grpc::Status;
using computenode::NewIndexRequest;
using computenode::ExistingIndexRequest;
using computenode::IndexCreationReply;
using computenode::SearchIndexRequest;
using computenode::SearchIndexReply;
using computenode::HeartbeatRequest;
using computenode::HeartbeatReply;
using google::protobuf::Empty;

ComputeClient::ComputeClient(std::string compute_address, int query_timeout) : query_timeout_(query_timeout) {
    // Create a gRPC channel to the compute node
    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(compute_address, grpc::InsecureChannelCredentials());
    stub_ = ComputeNode::NewStub(channel);
}

int ComputeClient::create_new_index(int d, int num_vectors, int num_partitions, bool store_index_locally, int num_search_workers) { 
    grpc::ClientContext context;

    // Set the request parameters
    NewIndexRequest create_request;
    create_request.set_vector_dimension(d);
    create_request.set_num_vectors(num_vectors);
    create_request.set_num_clusters(num_partitions); 
    create_request.set_store_index_locally(store_index_locally);
    create_request.set_num_search_workers(num_search_workers);

    // Make the request
    IndexCreationReply create_response; 
    grpc::Status status = stub_->CreateNewIndex(&context, create_request, &create_response);
    assert(status.ok());
    return create_response.index_id();
}

// Helper method to uri encode a file path
std::string uri_encode(std::string file_path) {
    std::filesystem::path absolute_path = std::filesystem::absolute(file_path);
    std::ostringstream oss;
    for (unsigned char c : absolute_path.string()) {
        // Unreserved characters and '/'
        if ((isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || c == '/')) {
            oss << c;
        } else {
            oss << '%' << std::uppercase << std::hex << int(c);
        }
    }
    return "file://" + oss.str();
}

int ComputeClient::load_existing_index(std::string file_path, bool store_index_locally, int num_search_workers, bool store_index_on_disk) { 
    std::string file_uri = uri_encode(file_path);
    
    // Set the request parameter
    grpc::ClientContext context;
    ExistingIndexRequest create_request;
    create_request.set_index_uri(file_uri);
    create_request.set_store_index_locally(store_index_locally);
    create_request.set_num_search_workers(num_search_workers);
    create_request.set_store_index_on_disk(store_index_on_disk);

    // Make the request
    IndexCreationReply create_response; 
    grpc::Status status = stub_->LoadExistingIndex(&context, create_request, &create_response);
    assert(status.ok());
    return create_response.index_id();
}


std::shared_ptr<SearchIndexResult> ComputeClient::search_index(int index_id, torch::Tensor search_vectors, int k, int nprobe, float recall_target) { 
    // Create the request
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(query_timeout_));

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
    auto rpc_start_time = std::chrono::high_resolution_clock::now();
    grpc::Status status = stub_->SearchIndex(&context, search_request, &search_response);
    auto rpc_end_time = std::chrono::high_resolution_clock::now();

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

    // Also record the rpc time
    result->total_query_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(rpc_end_time - rpc_start_time).count();
    return result;
}

int64_t ComputeClient::heartbeat(bool val_to_send) { 
    // Create the request
    grpc::ClientContext context;
    HeartbeatRequest heartbeat_request;
    heartbeat_request.set_value_to_return(val_to_send);

    // Make the request and return the time taken by the request
    HeartbeatReply heartbeat_response;
    auto rpc_start_time = std::chrono::high_resolution_clock::now();
    grpc::Status status = stub_->Heartbeat(&context, heartbeat_request, &heartbeat_response);
    auto rpc_end_time = std::chrono::high_resolution_clock::now();

    assert(status.ok());
    return std::chrono::duration_cast<std::chrono::nanoseconds>(rpc_end_time - rpc_start_time).count();
}

void ComputeClient::print_metrics() { 
    grpc::ClientContext context;
    Empty request;
    Empty response;
    stub_->PrintAndResetMetrics(&context, request, &response);
}