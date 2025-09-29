#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "storage_node.grpc.pb.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"

#include <iostream>
#include <memory>
#include <string>
#include <cassert>
#include <unordered_map>
#include <mutex>

#include <communication/coordinator_client.h>
#include <partitions/index_partition.h>
#include <partitions/partition_store.h>
#include <partitions/disk_arrow_index_partition.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using google::protobuf::Empty;

using storagenode::StorageNode;
using storagenode::RegisterPartitionRequest;
using storagenode::AddVectorRequest;
using storagenode::PerformSearchRequest;
using storagenode::PerformSearchResponse;

ABSL_FLAG(uint32_t, port, 8001, "Port to launch this service on");
ABSL_FLAG(std::string, coordinator_address, "localhost:5051", "Address of the coordinator service");

// Service implementation
class StorageServiceImpl final : public StorageNode::Service { 
public:
    StorageServiceImpl() { 
        partition_store_ = std::make_shared<PartitionStore>();
        partition_store_->initialize_workers();
    }

    Status AddNewPartition(ServerContext* context,
                        const RegisterPartitionRequest* request,
                        Empty* response) override {          
        
        try { 
            partition_store_->add_partition(request->partition_id(), request->code_size());
            std::cout << "AddNewPartiton called for partition " << request->partition_id() << std::endl;
        } catch(...) { 
            std::cout << "AddNewPartition caused an error" << std::endl;
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Failed to add the the specified partition");
        }
        
        return Status::OK;
    }

    Status AddVectors(ServerContext* context, const AddVectorRequest* request, Empty* response) override { 
        std::cout << "AddVectors called" << std::endl << std::flush;
        try { 
            // First verify the input parameters
            size_t partition_id = request->partition_id();
            int64_t num_vectors = request->num_vectors();

            auto vector_ids_field = request->vector_ids();
            if(vector_ids_field.size() != num_vectors) { 
                std::cout << "Storage Node: Expected Num Vector Ids - " << num_vectors << ", Actual - " << vector_ids_field.size() << std::endl;
                throw std::runtime_error("Invalid number of vector ids");
            }

            const auto& vector_values_field = request->vector_values();
            int64_t expected_vector_values =  partition_store_->get_dimension(partition_id) * num_vectors;
            if(vector_values_field.size() != expected_vector_values) { 
                std::cout << "Storage Node: Expected Num Vector Values - " << expected_vector_values << ", Actual - " << vector_values_field.size() << std::endl;
                throw std::runtime_error("Invalid number of vector values");
            }

            // Now add in the vectors into the partition store
            std::cout << "Calling partition store add store with vector values with " << vector_values_field.size() << " floats for partition " << partition_id << std::endl;
            partition_store_->add_vectors(partition_id, num_vectors, vector_ids_field.data(), vector_values_field.data());
        } catch(...) { 
            std::cout << "AddVectors returning INVALID_ARGUMENT" << std::endl;
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Failed to add vectors to the specified partition");
        }

        std::cout << "AddVectors returning OK" << std::endl << std::flush;
        return Status::OK;
    }

    Status PerformSearch(ServerContext* context, const PerformSearchRequest* request, PerformSearchResponse* response) override { 
        try { 
            // Load the input values
            size_t partition_id = request->partition_id();
            size_t k = request->k();
            int num_queries = request->num_queries();

            const auto& query_vectors_field = request->query_vectors();
            int64_t expected_vector_values =  partition_store_->get_dimension(partition_id) * num_queries;
            if(query_vectors_field.size() != expected_vector_values) { 
                throw std::runtime_error("Invalid number of vector values");
            }
            const float* query_vectors_ptr = reinterpret_cast<const float*>(query_vectors_field.data());

            // Determine the metric to use
            MetricType metric;
            switch(request->distance_metric()) { 
                case PerformSearchRequest::INNER_PRODUCT:
                    metric = faiss::METRIC_INNER_PRODUCT;
                    break;
                default:
                    metric = faiss::METRIC_L2;
                    break;
            }

            // Also get the fields to write the results
            int num_responses = num_queries * k;
            auto* result_ids_field = response->mutable_vector_ids();
            result_ids_field->Clear(); result_ids_field->Resize(num_responses, -1);
            int64_t* result_ids_ptr = reinterpret_cast<int64_t*>(result_ids_field->mutable_data());

            auto* result_distances_field = response->mutable_vector_distances();
            result_distances_field->Clear(); result_distances_field->Resize(num_responses, 0);
            float* result_distances_ptr = reinterpret_cast<float*>(result_distances_field->mutable_data());

            // Now actually perform the search
            std::cout << "[StorageNode] Calling perform search with args: K = " << k << ", Partition Id = " << partition_id << std::endl;
            partition_store_->perform_search(partition_id, k, num_queries, query_vectors_ptr, metric, result_ids_ptr, result_distances_ptr);

            std::cout << "[StorageNode] Returning result for search of - ";
            for(size_t i = 0; i < k; i++) { 
                std::cout << "(" << result_ids_field->Get(i) << "," << result_distances_field->Get(i) << "); ";
            }
            std::cout << std::endl;
        } catch(...) { 
            std::cout << "Returning invalid argument for PerformSearch" << std::endl;
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Failed to perform search");
        }

        std::cout << "[StorageNode] Returning OK for PerformSearch with fields of size: " << response->vector_ids().size() << ", " << response->vector_distances().size() << std::endl;
        return Status::OK;
    }

private:
    std::shared_ptr<PartitionStore> partition_store_; // Data structure storing the partitions
};

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);

    // First register the server with the coordinator
    std::string coordinator_address = absl::GetFlag(FLAGS_coordinator_address);
    std::shared_ptr<CoordinatorClient> coordinator_client = std::make_shared<CoordinatorClient>(coordinator_address);
    CoordinatorClient::SetGetCoordinatorClient(coordinator_client);

    uint16_t service_port = absl::GetFlag(FLAGS_port);
    bool registered_worker = coordinator_client->register_worker(service_port, false);
    assert(registered_worker && "Failed to register storage node with coordinator");

    // Now launch the grpc service for this storage node
    std::string server_address = absl::StrFormat("0.0.0.0:%d", service_port);
    StorageServiceImpl service;
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Starting Storage Node at " << server_address << std::endl;
    server->Wait(); 

    return 0;
}