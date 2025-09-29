#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <iostream>
#include <memory>
#include <string>
#include <set>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <cstdio>

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
using coordinator::RegisterIndexRequest;
using coordinator::RegisterIndexReply;

ABSL_FLAG(uint32_t, port, 5051, "Port to launch this service on");

std::string MergeAddress(const std::string& host, uint16_t port) {
  std::string host_without_port = host.substr(0, host.find_last_of(':'));
  return absl::StrFormat("%s:%d", host_without_port, port);
}

// Structs for storing various system state
struct Partition { 
  int partition_id_;
  int index_id_;

  Partition(int partition_id, int index_id) : partition_id_(partition_id), index_id_(index_id) { 

  }
}; 

struct Index { 
  std::string compute_id_; // The id of the compute node that owns this index
  int index_id_; 
  std::unordered_map<int, std::shared_ptr<Partition>> partitions_; // The partitions in this index

  Index(std::string compute_id, int index_id) : compute_id_(compute_id), index_id_(index_id) { 

  }
};

struct ComputeNode { 
  std::string compute_endpoint_; // The endpoint to use to communicate with the compute node
  std::vector<std::shared_ptr<Index>> indexes_;

  ComputeNode(std::string compute_endpoint) : compute_endpoint_(compute_endpoint) { 

  }
};

struct StorageNode { 
  std::string storage_address_;
  std::vector<std::shared_ptr<Partition>> partitions_; // The partitions that live in this storage node

  StorageNode(std::string storage_address) : storage_address_(storage_address) { 

  }
};

// Service implementation
class CoordinatorServiceImpl final : public Coordinator::Service {
public:

  Status RegisterCompute(ServerContext* context,
                        const RegisterComputeRequest* request,
                        Empty* response) override {

    // Save the compute worker in the state map
    std::lock_guard<std::mutex> lock(state_mutex_);
    std::string compute_address = context->peer();
    std::string compute_endpoint = MergeAddress(compute_address, request->service_port());

    compute_workers_[compute_address] = std::make_shared<ComputeNode>(compute_endpoint);
    std::cout << "Registered compute node with address " << compute_endpoint << std::endl;
    return Status::OK;
  }

  Status RegisterStorage(ServerContext* context,
                        const RegisterStorageRequest* request,
                        Empty* response) override {
    
    // Save the address of the storage worker
    std::lock_guard<std::mutex> lock(state_mutex_);
    std::string storage_address = context->peer();
    std::string storage_endpoint = MergeAddress(storage_address, request->service_port());

    storage_workers_[storage_address] = std::make_shared<StorageNode>(storage_endpoint);
    std::cout << "Registered storage node with address " << storage_endpoint << std::endl;
    return Status::OK;
  }

  Status RegisterNewIndex(ServerContext* context, const RegisterIndexRequest* request, RegisterIndexReply* response) { 
    std::lock_guard<std::mutex> lock(state_mutex_);
    std::cout << "RegisterNewIndex called with " << compute_workers_.size() << " computes and " << storage_workers_.size() << " storage nodes" << std::endl;

    // Verify that this compute node is registered
    std::string compute_address = context->peer();
    if(compute_workers_.find(compute_address) == compute_workers_.end()) { 
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "Unregistered compute node trying to create a new index");
    }
    std::shared_ptr<ComputeNode> compute_node = compute_workers_[compute_address];

    // First generate a new id for this index (don't need atomic since we acquire a global lock)
    int index_id = new_index_id_counter_; new_index_id_counter_ += 1;
    response->set_global_index_id(index_id);
    std::shared_ptr<Index> index = std::make_shared<Index>(compute_address, index_id);
    compute_node->indexes_.push_back(index);

    // Now determine the storage nodes to store the compute's partitions and write it out to the results
    std::vector<std::string> storage_node_keys;
    for(auto& curr_worker : storage_workers_) { 
      storage_node_keys.push_back(curr_worker.first);
    }
    int num_storage_nodes = storage_node_keys.size();

    int num_partitions = request->num_partitions();
    auto* result_partition_field = response->mutable_partition_storage_nodes();
    result_partition_field->Clear(); result_partition_field->Reserve(num_partitions);

    auto* result_partition_ids_field = response->mutable_partition_ids();
    result_partition_ids_field->Clear(); result_partition_ids_field->Reserve(num_partitions);

    for(int partition_id = 0; partition_id < num_partitions; partition_id++) { 
      std::shared_ptr<Partition> curr_partition = std::make_shared<Partition>(partition_id, index_id);
      index->partitions_[partition_id] = curr_partition;
      std::string partition_storage_key = storage_node_keys[partition_id % num_storage_nodes];

      storage_workers_[partition_storage_key]->partitions_.push_back(curr_partition);
      std::string storage_node_endpoint = storage_workers_[partition_storage_key]->storage_address_;
      result_partition_field->Add(std::string(storage_node_endpoint));
      std::cout << "For partition " << new_partition_id_counter_ << " determined storage node of " << storage_node_endpoint << std::endl; 

      result_partition_ids_field->Add(new_partition_id_counter_); new_partition_id_counter_++; 
    }
    std::cout << "Generated new index id of " << index_id << " for index with " << num_partitions << " partitions" << std::endl;

    return Status::OK;
  }

private:
  int new_index_id_counter_{0}; // Counter used to generate new index ids
  size_t new_partition_id_counter_{0}; // Counter used to generate globally unique partition ids
  std::unordered_map<std::string, std::shared_ptr<ComputeNode>> compute_workers_; // All the compute workers in the system
  std::unordered_map<std::string, std::shared_ptr<StorageNode>> storage_workers_; // All the storage workers in the system
  std::mutex state_mutex_; // Mutex that needs to be acquired to update coordinator state
};

int main(int argc, char** argv) {
    std::setbuf(stdout, nullptr);
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