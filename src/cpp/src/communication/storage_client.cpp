//
// Created by Devesh on 09/28/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include <communication/storage_client.h>

#include <cassert>

using grpc::Status;
using google::protobuf::Empty;
using storagenode::RegisterPartitionRequest;
using storagenode::AddVectorRequest;
using storagenode::PerformSearchRequest;
using storagenode::PerformSearchResponse;

StorageClient::StorageClient(std::string storage_address) { 
    // Create a gRPC channel to the storage node
    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(storage_address, grpc::InsecureChannelCredentials());
    stub_ = StorageNode::NewStub(channel);
    std::cout << "Created storage client to node " << storage_address << std::endl;
}

bool StorageClient::add_partition(size_t partition_id, int64_t code_size) { 
    grpc::ClientContext context;

    // Set the request parameters
    RegisterPartitionRequest register_request; 
    register_request.set_partition_id(partition_id);
    register_request.set_code_size(code_size);

    // Make the request
    Empty register_response;
    grpc::Status status = stub_->AddNewPartition(&context, register_request, &register_response);
    if(!status.ok()) { 
        std::cout << "[StorageClient] AddNewPartition failed with error code " << status.error_code() << " and error details of " << status.error_message() << std::endl;
    }

    return status.ok();
}

bool StorageClient::add_vectors(size_t partition_id, size_t num_vectors, size_t vector_dimension, const idx_t* ids, const float* vectors) { 
    // Create the request
    grpc::ClientContext context;

    AddVectorRequest add_request;
    add_request.set_partition_id(partition_id); add_request.set_num_vectors(num_vectors);

    auto* vector_ids_field = add_request.mutable_vector_ids();
    vector_ids_field->Clear(); vector_ids_field->Reserve(num_vectors);
    for(size_t i = 0; i < num_vectors; i++) { 
        vector_ids_field->Add(ids[i]);
    }

    size_t total_vector_values = num_vectors * vector_dimension;
    auto* vector_values_field = add_request.mutable_vector_values();
    vector_values_field->Clear(); vector_values_field->Reserve(total_vector_values);
    vector_values_field->Add(vectors, vectors + total_vector_values); 

    // Make the request
    Empty add_response;
    grpc::Status status = stub_->AddVectors(&context, add_request, &add_response);
    if(!status.ok()) { 
        std::cout << "[StorageClient] AddVectors failed with error code " << status.error_code() << " and error details of " << status.error_message() << std::endl;
    }

    return status.ok();
}

std::pair<std::vector<float>, std::vector<int64_t>> StorageClient::perform_search(size_t partition_id, size_t k, int num_queries, size_t vector_dimension, const float* query_vectors, MetricType metric) { 
    // Create the request
    grpc::ClientContext context;
    PerformSearchRequest search_request; 

    search_request.set_partition_id(partition_id); search_request.set_k(k);
    search_request.set_num_queries(num_queries);
    switch(metric) { 
        case faiss::METRIC_INNER_PRODUCT:
            search_request.set_distance_metric(PerformSearchRequest::INNER_PRODUCT);
            break;
        default:
            search_request.set_distance_metric(PerformSearchRequest::L2);
            break;
    }

    // Also copy over the query vectors
    size_t total_vector_values = num_queries * vector_dimension;
    auto* query_vector_field = search_request.mutable_query_vectors();
    query_vector_field->Clear(); query_vector_field->Reserve(total_vector_values);
    query_vector_field->Add(query_vectors, query_vectors + total_vector_values);

    // Make the request
    std::cout << "[Storage Client] Calling perform search for partition " << partition_id << std::endl;
    PerformSearchResponse search_response;
    grpc::Status status = stub_->PerformSearch(&context, search_request, &search_response);
    assert(status.ok());
    std::cout << "[Storage Client] Finished perform search for partition " << partition_id << " with okay status of " << status.ok() << std::endl;

    // Create the result vectors from the response
    int num_expected_responses = num_queries * k;
    std::vector<int64_t> result_ids;
    auto response_ids_field = search_response.vector_ids();
    assert(response_ids_field.size() == num_expected_responses);

    std::cout << "[Storage Client] Expected Responses - " << num_expected_responses << ", IDs Field Size - " << response_ids_field.size() << std::endl;
    for(int i = 0; i < num_expected_responses; i++) { 
        result_ids.push_back(response_ids_field.Get(i));
        std::cout << "[Storage Client] Result ID " << i << " has value of " << result_ids[result_ids.size() - 1] << std::endl;
    }
    std::cout << "[Storage Client] Result IDs Size - " << result_ids.size() << std::endl;

    std::vector<float> result_distances;
    auto response_distances_field = search_response.vector_distances();
    assert(response_distances_field.size() == num_expected_responses);

    std::cout << "[Storage Client] Expected Responses - " << num_expected_responses << ", Dist Field Size - " << response_distances_field.size() << std::endl;
    for(int i = 0; i < num_expected_responses; i++) { 
        result_distances.push_back(response_distances_field.Get(i));
        std::cout << "[Storage Client] Result Distance " << i << " has value of " << result_distances[result_distances.size() - 1] << std::endl;
    }
    std::cout << "[Storage Client] Result Distances Size - " << result_distances.size() << std::endl;

    return std::make_pair(result_distances, result_ids);
}