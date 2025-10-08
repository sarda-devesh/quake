//
// Created by Devesh on 09/25/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef REMOTE_INDEX_PARTITION_H
#define REMOTE_INDEX_PARTITION_H

#include <common.h>
#include <partitions/index_partition.h>
#include <communication/storage_client.h>

/**
 * @brief Represents a partition of encoded vectors that are stored on a storage node
 *
 * The RemoteIndexPartition is a child class of the IndexPartition are not stored in the current node
 * but in another node in the system. This is primarily used by the Compute Nodes to represent partitions
 * that are not stored in this node but rather in a storage node. The RemoteIndexPartition serves as a shimmy
 * layer that relays all requests to the appropriate storage node using the communication libraries. 
 */
class RemoteIndexPartition : public IndexPartition {
public:
    
    /// Default constructor
    RemoteIndexPartition() = default;

    /**
     * @brief Parameterized constructor.
     *
     * Initializes an empty remote partition
     * 
     * @param partition_id The id associated with this partition
     * @param code_size Size of each code in bytes
     * @param intialize_parameters The parameters used to intiialize this RemoteIndexPartition 
    */
    RemoteIndexPartition(size_t partition_id, int64_t code_size, std::shared_ptr<PartitionInitializeParams> initialize_parameters);

    /// Destructor
    ~RemoteIndexPartition() = default;

    IndexPartitionType get_index_type() override { 
        return IndexPartitionType::Remote;
    }

    /** 
     * @brief Method to get the codes of the partition
     * 
     * @return A pointer to a in memory copy of the encoded vectors of this partition. Note that this pointer
     * is owned by the index and thus shouldn't be modified/freed by the caller.
    */
    const uint8_t* get_codes() override;

    /** 
     * @brief Method to get the ids of the partition
     * 
     * @return A pointer to the ids associated with the codes of this partition. Note that this pointer
     * is owned by the index and thus shouldn't be modified/freed by the caller. 
    */
    const idx_t* get_ids() override;

    /** 
     * @brief Helper method to get the number of vector
     * 
     * @return The number of vectors currently in this partition
    */
    int64_t get_num_vectors() override;

    /**
     * @brief Helper method to get the code size
     * 
     * @return The size in bytes of each vector code.
     */
    int64_t get_code_size() override;

    /**
     * @brief Set the code size.
     *
     * Must be called before any vectors are added. Throws an error if vectors already exist.
     *
     * @param code_size The size in bytes for each vector code.
     */
    void set_code_size(int64_t code_size);

    /**
     * @brief Set the id for this partition
     *
     * Used to assign/reassign ids to the current partition 
     * 
     * @param partition_id The id to assign to this partition 
     */
    void set_partition_id(size_t partition_id) override;

    /**
     * @brief Append new entries to the partition.
     *
     * Appends n_entry new vectors (codes and IDs) to the partition.
     *
     * @param n_entry Number of new vectors to append.
     * @param new_ids Pointer to the new vector IDs.
     * @param new_codes Pointer to the new encoded vectors.
     */
    void append(int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) override;

    /**
     * @brief Update existing entries.
     *
     * Overwrites n_entry entries starting from the given offset.
     *
     * @param offset The starting index of the update.
     * @param n_entry Number of entries to update.
     * @param new_ids Pointer to the new vector IDs.
     * @param new_codes Pointer to the new encoded vectors.
     */
    void update(int64_t offset, int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) override;


    /**
     * @brief Remove an entry from the partition.
     *
     * Removes the vector at the given index from the partition
     *
     * @param index Index of the vector to remove.
     */
    void remove(int64_t index) override;

    /**
     * @brief Resize the partition.
     *
     * Ensures that the partition has capacity to store at least new_capacity entries.
     * 
     * @param new_capacity The desired capacity (number of vectors).
     */
    void resize(int64_t new_capacity) override; 

    /**
     * @brief Clear the partition.
     *
     * Resets the partition to an empty state
     */
    void clear() override;

    /**
     * @brief Find the index of a vector by its ID.
     *
     * A method to get the index of the vector with the specified id in the partition
     *
     * @param id The vector ID to search for.
     * @return The index of the vector if found; -1 otherwise.
     */
    int64_t find_id(idx_t id) override;

    /**
     * @brief Perform a top k search on this partition
     * 
     * @param k The number of neighbors to return
     * @param num_queries The number of query vector
     * @param query_vectors A pointer to the query vector data
     * @param MetricType The distance metric to use 
    */
    std::pair<std::vector<float>, std::vector<int64_t>> get_top_k(size_t k, int num_queries, const float* query_vectors, MetricType metric);

private:
    std::shared_ptr<StorageClient> storage_client_; // The client that can be used to communicate with the storage node
    int64_t num_vectors_; // The number of vectors currently in the partition
    int64_t code_size_; // The size of each code in bytes
    size_t partition_id_; // The local id associated with this partition
    size_t global_partition_id_; // The global id associated with this partition
    std::shared_ptr<PartitionInitializeParams> intialize_parameters_; // The parameters used to initialize this partition
    static constexpr bool debug_ = false; ///< If true, print debug information.
};


#endif //REMOTE_INDEX_PARTITION_H