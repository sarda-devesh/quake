//
// Created by Devesh on 09/28/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef STORAGE_CLIENT_H
#define STORAGE_CLIENT_H

#include <grpcpp/grpcpp.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cassert>
#include <unordered_map>
#include <utility>

#include <common.h>

#include "storage_node.grpc.pb.h"

using storagenode::StorageNode;

/**
 * @brief Wrapper to interact with a storage node via gRPC.
 *
 * The StorageClient is a client-side wrapper that facilitates communication
 * with a storage node via gRPC. All communication with a storage node
 * should be done through this class.
 */
class StorageClient { 
public: 
    /** 
    * @brief Constructor for StorageClient
    * @param storage_address The address of the storage node we want to communicate with (e.g., "localhost:9091").
    */
    StorageClient(std::string storage_address);

    /**
     * @brief Method to add a new partititon to the storage node
     * 
     * @param partition_id The id of the partition to add
     * @param code_size code size of the vectors in this partition
     * 
     * @return Whether we were able to sucessfully add the partition or not
     */
    bool add_partition(size_t partition_id, int64_t code_size);

    /**
     * @brief Method to add a new partititon to the storage node
     *  
     * @param partition_id The id of the partition to add
     * @param num_vectors The number of vectors to add
     * @param vector_dimension The dimension of the vectors to add
     * @param ids The ids of the vectors to add
     * @param vectors The values of the vectors to add
     * 
     * @return Whether we were able to add vectors or not
    */  
    bool add_vectors(size_t partition_id, size_t num_vectors, size_t vector_dimension, const idx_t* ids, const float* vectors);

    /**
     * @brief Method to find the top k neighbors for vectors in a partition
     * 
     * @param partition_id The id of the partition to add
     * @param k The number of neighbors to return
     * @param num_queries The number of query vector
     * @param vector_dimension The dimension of the vectors to add
     * @param query_vectors A pointer to the query vector data
     * @param MetricType The distance metric to use 
     * 
     * @return The ids of the nearest neighbors and their distances for the search vector
    */
    std::pair<std::vector<float>, std::vector<int64_t>> perform_search(size_t partition_id, size_t k, int num_queries, size_t vector_dimension, 
        const float* query_vectors, MetricType metric);

private:
    std::unique_ptr<StorageNode::Stub> stub_; ///< gRPC stub for making calls to the storage node
};

/**
 * @brief Class to get the storage client to a particular storage node
 * 
 * The StorageClientStore is just a wrapper that is stores mappings of storage node -> storage client.
 * This is used to that components can share the Storage Client rather than component needing to create the same
 * connection to the same storage node. 
 */
class StorageClientStore { 
public: 
    static std::shared_ptr<StorageClient> GetStorageClient(std::string storage_node) { 
        std::lock_guard<std::mutex> lock(map_mutex_);
        if(storage_clients_.find(storage_node) == storage_clients_.end()) { 
            storage_clients_[storage_node] = std::make_shared<StorageClient>(storage_node);
        }
        return storage_clients_[storage_node];
    }

private:
    inline static std::unordered_map<std::string, std::shared_ptr<StorageClient>> storage_clients_;
    inline static std::mutex map_mutex_; // Mutex that needs to be acquired to update the map
};

#endif // STORAGE_CLIENT_H