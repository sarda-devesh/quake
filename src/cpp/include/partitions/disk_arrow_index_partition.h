//
// Created by Jason on 12/18/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef ON_DISK_ARROW_INDEX_PARTITION_H
#define ON_DISK_ARROW_INDEX_PARTITION_H

#include <partitions/index_partition.h>

#include <common.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <filesystem>
#include <sstream>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>

/**
 * @brief A struct that represents a version of the partition that is stored on disk. 
 */
struct DiskArrowPartitionVersion { 
    int64_t version_id_; ///< The id of this version of the partition
    int64_t version_timestamp_; ///< The timestamp that this partition was created (in ns since epoch)
    int64_t num_vectors_;   ///< The number of stored vectors in this version
    std::filesystem::path data_path_; ///< The path to the file storing the actual data for this version of the partition
};

/**
 * @brief Represents a partition of encoded vectors that are stored on disk in arrow format
 *
 * The OnDiskArrowIndexPartition is a child class of IndexPartition that manages a contiguous block of 
 * encoded vectors (codes) and their corresponding vector IDs that are stored on disk in arrow format. 
 * Any changes (insertions, deletes, updates) are made using copy on write semantics, which generates
 * a new version of the partition stored in a new location as by default arrow buffers are immutable.  
 */
class OnDiskArrowIndexPartition : public IndexPartition {
public:
    uint64_t ids_copy_capacity_; ///< The capacity, in bytes, of the buffer storing a copy of the codes
    size_t ids_copy_version_; ///< The version of the the ids we have copied
    idx_t* ids_copy_; ///< Pointer to a buffer used to store an in memory copy of a version of the ids

    uint64_t codes_copy_capacity_; ///< The capacity, in bytes, of the buffer storing a copy of the codes
    size_t codes_copy_version_; ///< The version of the ids we have copied
    uint8_t* codes_copy_; ///< Pointer to a buffer used to store an in memory copy of a version of the codes

    size_t partition_id_; ///< The id associated with this partition
    int64_t code_size_ = 0; ///< Size of each code in bytes
    std::vector<DiskArrowPartitionVersion> partition_versions_; ///< A vector storing the different versions of the partition
    
    /// Default constructor.
    OnDiskArrowIndexPartition() = default;

    /**
     * @brief Parameterized constructor.
     *
     * Initializes the partition with a given number of vectors and copies in the provided codes and IDs.
     *
     * @param partition_id The id associated with this partition
     * @param num_vectors The initial number of vectors.
     * @param codes Pointer to the buffer holding the encoded vectors.
     * @param ids Pointer to the vector IDs.
     * @param code_size Size of each code in bytes.
     */
    OnDiskArrowIndexPartition(size_t partition_id,
                   int64_t num_vectors,
                   uint8_t* codes,
                   idx_t* ids,
                   int64_t code_size);

    /// Destructor
    ~OnDiskArrowIndexPartition();

    IndexPartitionType get_index_type() override { 
        return IndexPartitionType::OnDiskArrow;
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
    void set_code_size(int64_t code_size) override { 
        code_size_ = code_size;
    }

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
     * @brief Saves a new version of the partition to disk 
     * 
     * This method first persists the new values of this partition by writing the vectors and ids associated with this
     * partition to disk and then records this as a new version of the current partition
     * 
     * @param codes Pointer to the buffer holding the encoded vectors
     * @param ids Pointer to the vector IDs
     * @param num_vectors The number of vectors in this partition
     */
    void add_new_partition_version(const uint8_t* codes, const idx_t* ids, int64_t num_vectors);

    /**
     * @brief Writes the specified buffer to disk
     * 
     * This method writes the data in the provided buffer to disk by first generating a arrowArray 
     * representing the buffer in arrow format and then serializing the array into arrow's IPC
     * format and then writing the serialized format to disk 
     * 
     * @param buffer A pointer to the records that we want to write
     * @param num_bytes The size of the buffer in bytes
     * @param save_path The path where we want to save the buffer to
     */
    void write_buffer_to_disk(const uint8_t* buffer, int64_t buffer_size, std::filesystem::path save_path);

    /**
     * @brief Reads a buffer written to disk back into memory 
     * 
     * This method is used to read buffers written to disk using write_buffer_to_disk back into memory. Specifically,
     * it first reads the buffer from disk in an serialized format and then deserializes it and copies the raw values
     * back into the specified buffer. The method only copies the initial std::min(col_size, buffer_size) bytes and return
     * this value. 
     * 
     * @param data_path The path on disk from which we want to load the data
     * @param buffer A pointer to an in memory buffer where we want to load the saved records
     * @param buffer_size The size of the buffer we want to copy the records to
     * 
     * @return The number of the bytes in the input buffer that were actually copied
     */
    int64_t read_buffer_from_disk(std::filesystem::path data_path, uint8_t* buffer, int64_t buffer_size);
};

#endif //ON_DISK_ARROW_INDEX_PARTITION_H