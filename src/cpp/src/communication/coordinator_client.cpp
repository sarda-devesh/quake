//
// Created by Jason on 9/4/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include "communication/coordinator_client.h"

#include <cassert>

using grpc::Status;
using google::protobuf::Empty;
using coordinator::RegisterComputeRequest;
using coordinator::RegisterStorageRequest;
using coordinator::RegisterIndexRequest;
using coordinator::RegisterIndexReply;

CoordinatorClient::CoordinatorClient(std::string coordinator_address) {
    // Create a gRPC channel to the coordinator service
    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(coordinator_address, grpc::InsecureChannelCredentials());
    stub_ = Coordinator::NewStub(channel);
}

bool CoordinatorClient::register_worker(uint32_t service_port, bool is_compute_worker) {
    grpc::ClientContext context;
    Empty response;

    // Depending on the type of worker, call the appropriate registration method
    bool result_value = false;
    if (is_compute_worker) {
        RegisterComputeRequest request;
        request.set_service_port(service_port);
        grpc::Status status = stub_->RegisterCompute(&context, request, &response);
        result_value = status.ok();
    } else {
        RegisterStorageRequest request;
        request.set_service_port(service_port);
        grpc::Status status = stub_->RegisterStorage(&context, request, &response);
        result_value = status.ok();
    }

    return result_value;
}

std::shared_ptr<DistributedIndexDetails> CoordinatorClient::register_new_index(int num_partitions) { 
    grpc::ClientContext context;

    // Send the request for the new index
    RegisterIndexRequest index_request; 
    index_request.set_num_partitions(num_partitions);
    RegisterIndexReply index_response;
    grpc::Status status = stub_->RegisterNewIndex(&context, index_request, &index_response);
    if(!status.ok()) { 
        std::cout << "[Coordinate Client] RegisterNewIndex failed with error code of " << status.error_code() << " and error details of " << status.error_details() << std::endl;
    }
    assert(status.ok()); 

    // Populate the new index details from the response
    std::shared_ptr<DistributedIndexDetails> new_index_details = std::make_shared<DistributedIndexDetails>();
    new_index_details->index_id = index_response.global_index_id();

    auto partition_nodes_field = index_response.partition_storage_nodes();
    auto partition_ids_field = index_response.partition_ids();
    assert(partition_nodes_field.size() == num_partitions); assert(partition_ids_field.size() == num_partitions);
    for(int i = 0; i < num_partitions; i++) { 
        std::string curr_partition_node = partition_nodes_field.Get(i);
        new_index_details->partition_storage_nodes.push_back(curr_partition_node);
        size_t curr_partition_id = partition_ids_field.Get(i);
        new_index_details->partition_ids.push_back(curr_partition_id);
    }

    return new_index_details;
}