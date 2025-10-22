#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "compute_node.grpc.pb.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"

#include <memory>
#include <atomic>
#include <unordered_map>
#include <cassert>
#include <filesystem>
#include <chrono>

#include <communication/coordinator_client.h>
#include <quake_index.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using google::protobuf::Empty;

using computenode::ComputeNode;
using computenode::NewIndexRequest;
using computenode::ExistingIndexRequest;
using computenode::IndexCreationReply;
using computenode::SearchIndexRequest;
using computenode::SearchIndexReply;
using computenode::HeartbeatRequest;
using computenode::HeartbeatReply;

ABSL_FLAG(uint32_t, port, 9001, "Port to launch this service on");
ABSL_FLAG(std::string, coordinator_address, "localhost:5051", "Address of the coordinator service");

// Helper method to decode the file path from the uri
std::string uri_decode(std::string encoded) {
    std::string file_uri_prefix = "file://";
    encoded = encoded.substr(file_uri_prefix.length());

    std::ostringstream oss;
    for (size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] == '%' && i + 2 < encoded.size() &&
            std::isxdigit(encoded[i+1]) && std::isxdigit(encoded[i+2])) {
            // decode %XX hex
            std::string hex = encoded.substr(i+1, 2);
            oss << static_cast<char>(std::stoi(hex, nullptr, 16));
            i += 2;
        } else {
            oss << encoded[i];
        }
    }
    return oss.str();
}

class ComputeNodeServiceImpl final : public ComputeNode::Service {
public:
    Status CreateNewIndex(ServerContext* context,
                        const NewIndexRequest* request,
                        IndexCreationReply* response) override {
        
        // Determine the index details based on whether it is going to be stored in memory or remotely
        int num_clusters = request->num_clusters();
        std::shared_ptr<DistributedIndexDetails> new_index_details = nullptr; 
        int new_index_id = -1 * (indexes_.size() + 1); // -1 so that it doesn't conflict with an global index id
        if(!request->store_index_locally()) { 
            std::shared_ptr<CoordinatorClient> coordinator_client = CoordinatorClient::GetCoordinatorClient();
            new_index_details = coordinator_client->register_new_index(num_clusters);
            new_index_id = new_index_details->index_id;
        }

        // Create the new index
        std::shared_ptr<QuakeIndex> new_index = std::make_shared<QuakeIndex>(new_index_id);

        // Build the index based on the specified user parameters
        int vector_dimension = request->vector_dimension();
        int num_vectors = request->num_vectors();

        std::shared_ptr<IndexBuildParams> build_params = std::make_shared<IndexBuildParams>();
        build_params->dimension = vector_dimension;
        build_params->nlist = num_clusters;
        build_params->distributed_index_details = new_index_details;
        build_params->num_workers = request->num_search_workers();
        
        torch::Tensor build_vectors_ = torch::randn({num_vectors, vector_dimension}, torch::kFloat32);
        torch::Tensor build_ids_ = torch::arange(0, num_vectors, torch::kInt64);
        std::cout << "[Compute Node] Building index " << new_index_id << " with details: D - " << vector_dimension << ", # vectors - " << num_vectors << ", nlist - " << num_clusters << std::endl << std::flush;
        new_index->build(build_vectors_, build_ids_, build_params);

        // Save the new index
        indexes_[new_index_id] = new_index; 
        response->set_index_id(new_index_id);
        return Status::OK;
    }

    Status LoadExistingIndex(ServerContext* context,
                        const ExistingIndexRequest* request,
                        IndexCreationReply* response) { 
        
        // Extract the index path from the uri
        std::string index_dir_str = uri_decode(request->index_uri());
        std::shared_ptr<QuakeIndex> new_index = nullptr; 
        
        // Load the index
        int default_index_id = -1 * (indexes_.size() + 1); // This will get overriden by the global index id if we decide to distribute the index
        new_index = std::make_shared<QuakeIndex>(default_index_id); 
        new_index->load(index_dir_str, request->num_search_workers(), !request->store_index_locally(), request->store_index_on_disk());

        // Save the new index
        int new_index_id = new_index->index_id_;
        response->set_index_id(new_index_id);
        indexes_[new_index_id] = new_index; 
        if constexpr(debug_) std::cout << "Loaded index from path " << index_dir_str << " with index id of " << new_index_id << std::endl;

        return Status::OK;
    }

    void add_metric_to_map(std::string metric_name, float metric_value) { 
        if(metrics_.find(metric_name) == metrics_.end()) { 
            metrics_[metric_name] = std::make_shared<MetricStore>(metric_name);
        }
        metrics_[metric_name]->add_value(metric_value);
    }

    void record_search_metrics(std::shared_ptr<SearchResult> search_result) {
        std::shared_ptr<SearchTimingInfo> partition_scan_info = search_result->timing_info;
        std::shared_ptr<SearchTimingInfo> centroid_scan_info = partition_scan_info->parent_info;

        // Add in the end to end times
        int64_t total_search_time_ns = partition_scan_info->total_time_ns;
        int64_t centroid_search_time_ns = centroid_scan_info->total_time_ns;
        int64_t partition_search_time_ns = total_search_time_ns - centroid_search_time_ns;

        add_metric_to_map("centorid_search_time_ms", centroid_search_time_ns/MS_TO_NS);
        add_metric_to_map("partition_search_time_ms", partition_search_time_ns/MS_TO_NS);
        add_metric_to_map("grpc_result_write_time_ms", search_result->result_write_time_ns/MS_TO_NS);

        // Add in partition scan breakdown
        add_metric_to_map("partition_search_job_enque_time_ms", partition_scan_info->job_enqueue_time_ns/MS_TO_NS);
        add_metric_to_map("partition_search_job_wait_time_ms", partition_scan_info->job_wait_time_ns/MS_TO_NS);
        add_metric_to_map("partition_search_result_aggregate_time_ms", partition_scan_info->result_aggregate_time_ns/MS_TO_NS);
        
        // Add in centroid scan breakdown

    }

    Status SearchIndex(ServerContext* context,
                        const SearchIndexRequest* request,
                        SearchIndexReply* response) override {
        
        // Verify the query parameters
        int index_id = request->index_id();
        if(indexes_.find(index_id) == indexes_.end()) {
            std::cout << "SearchIndex called for invalid index id of " << index_id << std::endl; 
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid Index Id");
        }

        std::shared_ptr<QuakeIndex> search_index_ = indexes_[index_id];
        int vector_dimension = search_index_->d();
        auto search_vector_field = request->search_vectors();
        if(search_vector_field.size() % vector_dimension != 0) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid Search Vector");
        }

        // Construct the search parameters based on the user input
        int num_queries = search_vector_field.size()/vector_dimension;
        torch::Tensor vectors = torch::from_blob(const_cast<float*>(search_vector_field.data()), {num_queries, vector_dimension}, torch::kFloat32);
        
        std::shared_ptr<SearchParams> search_params = std::make_shared<SearchParams>();
        search_params->k = request->top_k();
        search_params->nprobe = request->num_probe();
        search_params->recall_target = request->recall_target();

        // Actually perform the search and save the result in the response
        try { 
            if constexpr(debug_) { 
                std::cout << "[Compute Node] Performing search with details: Index - " << index_id << ", Num Queries - " << num_queries << ", K - " << search_params->k << ", NProbe - " << search_params->nprobe << ", Recall Target - " << search_params->recall_target << std::endl;
            }
            std::shared_ptr<SearchResult> search_result = search_index_->search(vectors, search_params);
            if constexpr(debug_) {
                std::cout << "Finished search in " << search_result->timing_info->total_time_ns << " ns" << std::endl;
            }

            auto result_write_start = std::chrono::high_resolution_clock::now();
            int num_responses = num_queries * request->top_k();
            torch::Tensor response_ids = torch::flatten(search_result->ids).to(torch::kInt64);
            int64_t* ids_ptr = response_ids.data_ptr<int64_t>();
            auto* result_ids_field = response->mutable_result_ids();
            result_ids_field->Clear(); result_ids_field->Reserve(num_responses);
            result_ids_field->Add(ids_ptr, ids_ptr + num_responses);

            torch::Tensor response_distances = torch::flatten(search_result->distances).to(torch::kFloat);
            float* distances_ptr = response_distances.data_ptr<float>();
            auto* result_distances_field = response->mutable_result_distances();
            result_distances_field->Clear(); result_distances_field->Reserve(num_responses);
            result_distances_field->Add(distances_ptr, distances_ptr + num_responses);
            auto result_write_end = std::chrono::high_resolution_clock::now();

            search_result->result_write_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(result_write_end - result_write_start).count();
            record_search_metrics(search_result);
        } catch (const std::exception& e) {
            // Catch any other standard exception type
            std::cerr << "Vector Search got exception of " << e.what() << std::endl;
            return grpc::Status(grpc::StatusCode::UNKNOWN, "Searching the Index resulted in an error");
        }
        

        return Status::OK;
    }

    Status Heartbeat(ServerContext* context,
                        const HeartbeatRequest* request,
                        HeartbeatReply* response) override { 
        
        response->set_reply_value(request->value_to_return());
        return Status::OK;
    }

    Status PrintAndResetMetrics(ServerContext* context, const Empty* request, Empty* response) override { 
        std::cout << "------- START COMPUTE NODE METRICS -----" << std::endl;
        // Print out compute node metrics
        for(const auto& pair : metrics_) { 
            pair.second->print_metric();
            pair.second->reset_metric();
        }

        // Print per worker metrics
        for(const auto& index_pair : indexes_) { 
            int index_id = index_pair.first;
            auto& index_workers = index_pair.second->query_coordinator_->core_resources_;
            for(size_t i = 0; i < index_workers.size(); i++) { 
                auto& worker_metrics = index_workers[i].metrics;
                for(const auto& metric_pair : worker_metrics) { 
                    std::cout << "Index " << index_id << " Worker " << i << " ";
                    metric_pair.second->print_metric();
                    metric_pair.second->reset_metric();
                }
            }
        }
        std::cout << "------- FINISH COMPUTE NODE METRICS -----" << std::endl;
        std::cout << std::endl;

        // Now also call print metrics on the storage nodes
        for(const auto& storage_client_pair : StorageClientStore::storage_clients_) { 
            storage_client_pair.second->print_metrics();
        }
        return Status::OK;
    }

private:
    static constexpr bool debug_ = false; 
    std::unordered_map<int, std::shared_ptr<QuakeIndex>> indexes_; // Map storing all the indexes in this compute node
    std::unordered_map<std::string, std::shared_ptr<MetricStore>> metrics_; // Map of all of the compute node related metrics
};

int main(int argc, char** argv) {
    std::setbuf(stdout, nullptr);
    absl::ParseCommandLine(argc, argv);

    // First register the server with the coordinator
    std::string coordinator_address = absl::GetFlag(FLAGS_coordinator_address);
    std::shared_ptr<CoordinatorClient> coordinator_client = std::make_shared<CoordinatorClient>(coordinator_address);
    CoordinatorClient::SetGetCoordinatorClient(coordinator_client);

    uint16_t service_port = absl::GetFlag(FLAGS_port);
    bool registered_worker = coordinator_client->register_worker(service_port, true);
    assert(registered_worker && "Failed to register compute node with coordinator");

    // Now launch the grpc service for this compute node
    std::string server_address = absl::StrFormat("0.0.0.0:%d", service_port);
    ComputeNodeServiceImpl service;
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Starting Compute Node at " << server_address << std::endl;
    server->Wait();

    return 0;
}