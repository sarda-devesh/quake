#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <iostream>
#include <memory>
#include <string>
#include <set>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"
#include "coordinator.grpc.pb.h" 

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using google::protobuf::Empty;

using coordinator::Coordinator;
using coordinator::RegisterComputeRequest;
using coordinator::RegisterStorageRequest;

ABSL_FLAG(uint32_t, port, 5051, "Port to launch this service on");

// Class to store the current state of the coordinator service
class CoordinatorServiceState {
public:
    std::set<std::string> compute_workers;
    std::set<std::string> storage_workers;
};

std::string MergeAddress(const std::string& host, uint16_t port) {
  std::string host_without_port = host.substr(0, host.find_last_of(':'));
  return absl::StrFormat("%s:%d", host_without_port, port);
}

// Service implementation
class CoordinatorServiceImpl final : public Coordinator::Service {
public:
  CoordinatorServiceImpl() {

  }

  ~CoordinatorServiceImpl() { 

  }

  Status RegisterCompute(ServerContext* context,
                        const RegisterComputeRequest* request,
                        Empty* response) override {
    
    // Save the address of the compute worker
    std::string client_address = MergeAddress(context->peer(), request->service_port());
    service_state_.compute_workers.insert(client_address);
                    
    size_t num_total_computes = service_state_.compute_workers.size();
    std::cout << "Registered Compute " << client_address << " with a total of " << num_total_computes << " computes" << std::endl;
    return Status::OK;
  }

  Status RegisterStorage(ServerContext* context,
                        const RegisterStorageRequest* request,
                        Empty* response) override {
    
    // Save the address of the storage worker
    std::string client_address = MergeAddress(context->peer(), request->service_port());
    service_state_.storage_workers.insert(client_address);

    size_t num_total_storages = service_state_.storage_workers.size();
    std::cout << "Registered Storage " << client_address << " with a total of " << num_total_storages << " storage workers" << std::endl;
    return Status::OK;
  }

private:
    CoordinatorServiceState service_state_;
};

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);

    // Determine the server address based on the port
    uint16_t service_port = absl::GetFlag(FLAGS_port);
    std::string server_address = absl::StrFormat("0.0.0.0:%d", service_port);

    // Create the grpc service without any authentication
    CoordinatorServiceImpl service;
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    // Now start the server
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Coordinator listening on " << server_address << std::endl;
    server->Wait();

    return 0;
}