//
// Created by Jason on 12/18/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef IN_MEM_INDEX_PARTITION_H
#define IN_MEM_INDEX_PARTITION_H

#include <common.h>
#include <partitions/index_partition.h>

/**
 * @brief Represents a partition of encoded vectors that are stored in memory
 *
 * The InMemoryIndexPartition is a child class of IndexPartition that manages a contiguous block of 
 * encoded vectors (codes) and their corresponding vector IDs in memory. If QUAKE_USE_NUMA is defined,
 * it supports NUMA-aware memory allocation and reallocation.
 */
class InMemoryIndexPartition : public IndexPartition {
public:
    size_t partition_id_ = 0; ///< The id associated with the given partition
    int numa_node_ = -1;    ///< Assigned NUMA node (-1 if not set)
    int core_id_ = -1;    ///< Mapped thread ID for processing

    int64_t buffer_size_ = 0;   ///< Allocated capacity (in number of vectors)
    int64_t num_vectors_ = 0;   ///< Current number of stored vectors
    int64_t code_size_ = 0;     ///< Size of each code in bytes (must be set before adding vectors)

    uint8_t* codes_ = nullptr;  ///< Pointer to the encoded vectors (raw memory block)
    idx_t* ids_ = nullptr;      ///< Pointer to the vector IDs

    std::unordered_map<idx_t, int64_t> id_to_index_; ///< Map of vector ID to index

    /// Default constructor.
    InMemoryIndexPartition() = default;

    /**
     * @brief Parameterized constructor.
     *
     * Initializes the partition with a given number of vectors and copies in the provided codes and IDs.
     *
     * @param num_vectors The initial number of vectors.
     * @param codes Pointer to the buffer holding the encoded vectors.
     * @param ids Pointer to the vector IDs.
     * @param code_size Size of each code in bytes.
     */
    InMemoryIndexPartition(int64_t num_vectors,
                   uint8_t* codes,
                   idx_t* ids,
                   int64_t code_size);

    /**
     * @brief Move constructor.
     *
     * Transfers the contents from another partition into this one.
     *
     * @param other The partition to move from.
     */
    InMemoryIndexPartition(InMemoryIndexPartition&& other) noexcept;

    /**
     * @brief Move assignment operator.
     *
     * Transfers the contents from another partition into this one, clearing existing data.
     *
     * @param other The partition to move from.
     * @return Reference to this partition.
     */
    InMemoryIndexPartition& operator=(InMemoryIndexPartition&& other) noexcept;

    /// Destructor. Frees all allocated memory.
    ~InMemoryIndexPartition();

    IndexPartitionType get_index_type() override { 
        return IndexPartitionType::InMemory;
    }

    /** 
     * @brief Method to get the codes of the partition
     * 
     * @return A pointer to a in memory copy of the encoded vectors of this partition. Note that this pointer
     * is owned by the index and thus shouldn't be modified/freed by the caller. 
    */
    const uint8_t* get_codes() override { 
        return codes_;
    }

    /** 
     * @brief Method to get the ids of the partition
     * 
     * @return A pointer to the ids associated with the codes of this partition. Note that this pointer
     * is owned by the index and thus shouldn't be modified/freed by the caller. 
    */
    const idx_t* get_ids() override { 
        return ids_;
    }

    /** 
     * @brief Helper method to get the number of vector
     * 
     * @return The number of vectors currently in this partition
    */
    int64_t get_num_vectors() override { 
        return num_vectors_;
    }

    /**
     * @brief Set the id for this partition
     *
     * Used to assign/reassign ids to the current partition 
     * 
     * @param partition_id The id to assign to this partition 
     */
    void set_partition_id(size_t partition_id) override { 
        partition_id_ = partition_id;
    }

    /**
     * @brief Returns the code size
     * 
     * @returns The size in bytes of each vector code.
     */
    int64_t get_code_size() override { 
        return code_size_;
    }

    /**
     * @brief Set the code size.
     *
     * Must be called before any vectors are added. Throws an error if vectors already exist.
     *
     * @param code_size The size in bytes for each vector code.
     */
    void set_code_size(int64_t code_size) override;

    /**
     * @brief Append new entries to the partition.
     *
     * Appends n_entry new vectors (codes and IDs) at the end of the partition.
     *
     * @param n_entry Number of new vectors to append.
     * @param new_ids Pointer to the new vector IDs.
     * @param new_codes Pointer to the new encoded vectors.
     */
    void append(int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) override;

    /**
     * @brief Update existing entries in place.
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
     * Removes the vector at the given index by swapping in the last vector.
     *
     * @param index Index of the vector to remove.
     */
    void remove(int64_t index) override;

    /**
     * @brief Resize the partition.
     *
     * Ensures that the internal buffer has capacity for at least new_capacity entries.
     * If new_capacity is less than the current number of vectors, the partition is truncated.
     *
     * @param new_capacity The desired capacity (number of vectors).
     */
    void resize(int64_t new_capacity) override;

    /**
     * @brief Clear the partition.
     *
     * Frees all allocated memory and resets the partition state.
     */
    void clear() override;

    /**
     * @brief Find the index of a vector by its ID.
     *
     * Performs a linear search.
     *
     * @param id The vector ID to search for.
     * @return The index of the vector if found; -1 otherwise.
     */
    int64_t find_id(idx_t id) override;

    /**
     * @brief Reallocate internal memory to a new capacity.
     *
     * Allocates new memory for a given capacity and copies existing data.
     *
     * @param new_capacity The new capacity (number of vectors).
     */
    void reallocate_memory(int64_t new_capacity);

    void set_core_id(int core_id);

#ifdef QUAKE_USE_NUMA
    /**
     * @brief Set the NUMA node for the partition.
     *
     * Moves the memory to the specified NUMA node if necessary.
     *
     * @param new_numa_node The target NUMA node.
     */
    void set_numa_node(int new_numa_node);
#endif

private:
    /**
     * @brief Move data from another partition.
     *
     * Helper for move constructor and move assignment.
     *
     * @param other The partition to move from.
     */
    void move_from(InMemoryIndexPartition&& other);

    /**
     * @brief Free the allocated memory.
     *
     * Releases the codes and IDs buffers.
     */
    void free_memory();

    /**
     * @brief Ensure capacity.
     *
     * Checks that the internal buffer can hold at least the required number of vectors,
     * and resizes if necessary.
     *
     * @param required The minimum required number of vectors.
     */
    void ensure_capacity(int64_t required);

    /**
     * @brief Allocate memory for a given type.
     *
     * Allocates memory for num_elements of type T, optionally on a specific NUMA node.
     *
     * @tparam T The data type.
     * @param num_elements The number of elements to allocate.
     * @param numa_node The NUMA node (-1 for default allocation).
     * @return Pointer to the allocated memory.
     */
    template <typename T>
    T* allocate_memory(size_t num_elements, int numa_node);
};

#endif //IN_MEM_INDEX_PARTITION_H
