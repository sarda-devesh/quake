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

#include <communication/coordinator_client.h>
#include <quake_index.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using google::protobuf::Empty;

using computenode::ComputeNode;
using computenode::NewIndexRequest;
using computenode::NewIndexReply;
using computenode::SearchIndexRequest;
using computenode::SearchIndexReply;

ABSL_FLAG(uint32_t, port, 9001, "Port to launch this service on");
ABSL_FLAG(std::string, coordinator_address, "localhost:5051", "Address of the coordinator service");

class ComputeNodeServiceImpl final : public ComputeNode::Service {
public:
    ComputeNodeServiceImpl(std::shared_ptr<CoordinatorClient> coordinator_client) : coordinator_client_(coordinator_client) { 

    }

    Status CreateNewIndex(ServerContext* context,
                        const NewIndexRequest* request,
                        NewIndexReply* response) override {
        
        // Create the new index
        int new_index_id = index_id_generator_.fetch_add(1, std::memory_order_relaxed);;
        std::shared_ptr<QuakeIndex> new_index = std::make_shared<QuakeIndex>(new_index_id);

        // Build the index based on the specified user parameters
        int vector_dimension = request->vector_dimension();
        int num_vectors = request->num_vectors();
        int num_clusters = request->num_clusters();

        torch::Tensor build_vectors_ = torch::randn({num_vectors, vector_dimension}, torch::kFloat32);
        torch::Tensor build_ids_ = torch::arange(0, num_vectors, torch::kInt64);
        std::shared_ptr<IndexBuildParams> build_params = std::make_shared<IndexBuildParams>();
        build_params->dimension = vector_dimension;
        build_params->nlist = num_clusters;

        std::cout << "[Compute Node] Building index " << new_index_id << " with details: D - " << vector_dimension << ", # vectors - " << num_vectors << ", nlist - " << num_clusters << std::endl;
        new_index->build(build_vectors_, build_ids_, build_params);

        // Save the new index
        indexes_[new_index_id] = new_index; 
        response->set_index_id(new_index_id);
        return Status::OK;
    }

    Status SearchIndex(ServerContext* context,
                        const SearchIndexRequest* request,
                        SearchIndexReply* response) override {
        
        // Verify the query parameters
        int index_id = request->index_id();
        if(indexes_.find(index_id) == indexes_.end()) { 
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
        std::cout << "[Compute Node] Performing search with details: Index - " << index_id << ", Num Queries - " << num_queries << ", K - " << search_params->k << ", NProbe - " << search_params->nprobe << ", Recall Target - " << search_params->recall_target;
        std::shared_ptr<SearchResult> search_result = search_index_->search(vectors, search_params);
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

        return Status::OK;
    }

private:
    std::unordered_map<int, std::shared_ptr<QuakeIndex>> indexes_; // Map storing all the indexes managed by this compute node
    std::atomic<int> index_id_generator_{0}; // Atomic used to generate new index ids
    std::shared_ptr<CoordinatorClient> coordinator_client_; // Client that can be used to talk with the coordinator
};

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);

    // First register the server with the coordinator
    std::string coordinator_address = absl::GetFlag(FLAGS_coordinator_address);
    std::shared_ptr<CoordinatorClient> coordinator_client = std::make_shared<CoordinatorClient>(coordinator_address);

    uint16_t service_port = absl::GetFlag(FLAGS_port);
    bool registered_worker = coordinator_client->register_worker(service_port, true);
    assert(registered_worker && "Failed to register compute node with coordinator");

    // Now launch the grpc service for this compute node
    std::string server_address = absl::StrFormat("0.0.0.0:%d", service_port);
    ComputeNodeServiceImpl service(coordinator_client);
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Started Compute Node at " << server_address << std::endl;
    server->Wait();

    return 0;
}