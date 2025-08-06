//
// Created by Devesh on 08/05/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef INDEX_PARTITION_H
#define INDEX_PARTITION_H

#include <common.h>

/**
 * @brief An enum representing the different index partition types
 */
enum IndexPartitionType {
    InMemory, ///< In-memory index partition
};

/**
 * @brief Represents a partition (sub-index) of encoded vectors.
 *
 * The IndexPartition class manages a contiguous block of encoded vectors (codes)
 * and their corresponding vector IDs. It supports appending new entries, updating
 * and removing existing ones, and dynamically resizing the underlying memory.
 */
class IndexPartition {
public:
    /** 
     * @brief Returns the type of the current partition
     * 
     * @return An enum representing the type of the current partition
    */
    virtual IndexPartitionType get_index_type() = 0;

    /** 
     * @brief Method to get the codes of the partition
     * 
     * @return A pointer to a in memory copy of the encoded vectors of this partition. Note that this pointer
     * is owned by the index and thus shouldn't be modified/freed by the caller.
    */
    virtual const uint8_t* get_codes() = 0;

    /** 
     * @brief Method to get the ids of the partition
     * 
     * @return A pointer to the ids associated with the codes of this partition. Note that this pointer
     * is owned by the index and thus shouldn't be modified/freed by the caller. 
    */
    virtual const idx_t* get_ids() = 0;

    /** 
     * @brief Helper method to get the number of vector
     * 
     * @return The number of vectors currently in this partition
    */
    virtual int64_t get_num_vectors() = 0;

    /**
     * @brief Helper method to get the code size
     * 
     * @return The size in bytes of each vector code.
     */
    virtual int64_t get_code_size() = 0;

    /**
     * @brief Set the code size.
     *
     * Must be called before any vectors are added. Throws an error if vectors already exist.
     *
     * @param code_size The size in bytes for each vector code.
     */
    virtual void set_code_size(int64_t code_size) = 0;

    /**
     * @brief Append new entries to the partition.
     *
     * Appends n_entry new vectors (codes and IDs) to the partition.
     *
     * @param n_entry Number of new vectors to append.
     * @param new_ids Pointer to the new vector IDs.
     * @param new_codes Pointer to the new encoded vectors.
     */
    virtual void append(int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) = 0;

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
    virtual void update(int64_t offset, int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) = 0;

    /**
     * @brief Remove an entry from the partition.
     *
     * Removes the vector at the given index from the partition
     *
     * @param index Index of the vector to remove.
     */
    virtual void remove(int64_t index) = 0;

    /**
     * @brief Resize the partition.
     *
     * Ensures that the partition has capacity to store at least new_capacity entries.
     * 
     * @param new_capacity The desired capacity (number of vectors).
     */
    virtual void resize(int64_t new_capacity) = 0;

    /**
     * @brief Clear the partition.
     *
     * Resets the partition to an empty state
     */
    virtual void clear() = 0;

    /**
     * @brief Find the index of a vector by its ID.
     *
     * A method to get the index of the vector with the specified id in the partition
     *
     * @param id The vector ID to search for.
     * @return The index of the vector if found; -1 otherwise.
     */
    virtual int64_t find_id(idx_t id) const = 0;
}; 

#endif //INDEX_PARTITION_H