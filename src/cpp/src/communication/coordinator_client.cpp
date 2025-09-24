//
// Created by Jason on 9/4/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include "communication/coordinator_client.h"

using grpc::Status;
using google::protobuf::Empty;
using coordinator::RegisterComputeRequest;
using coordinator::RegisterStorageRequest;

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