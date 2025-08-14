// disk_arrow_index_partition.cpp

#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include <chrono>

#include "disk_arrow_index_partition.h"  // Include the OnDiskArrowIndexPartition header

using namespace faiss;

class OnDiskArrowIndexPartitionTest : public ::testing::Test {
protected:
    int64_t initial_num_vectors = 100;
    int64_t code_size = 128; // bytes per code
    shared_ptr<OnDiskArrowIndexPartition> partition;

    // Vectors to hold initial codes and ids for verification
    std::vector<uint8_t> curr_codes_vec_;
    std::vector<idx_t> curr_ids_vec_;

    virtual void SetUp() {
        // Initialize initial_ids with sequential IDs starting from 0
        generate_sequential_ids(initial_num_vectors, curr_ids_vec_, 0);

        // Initialize initial_codes with sequential codes starting from 100
        generate_sequential_codes(initial_num_vectors, curr_codes_vec_, 100);

        // Allocate memory and copy initial data
        uint8_t* initial_codes = static_cast<uint8_t*>(std::malloc(initial_num_vectors * code_size));
        idx_t* initial_ids = static_cast<idx_t*>(std::malloc(initial_num_vectors * sizeof(idx_t)));
        std::memcpy(initial_codes, curr_codes_vec_.data(), initial_num_vectors * code_size);
        std::memcpy(initial_ids, curr_ids_vec_.data(), initial_num_vectors * sizeof(idx_t));

        // Initialize an IndexPartition with initial data
        partition = make_shared<OnDiskArrowIndexPartition>(0, initial_num_vectors, initial_codes, initial_ids, code_size);

        // Free temporary allocations as IndexPartition has its own copies
        std::free(initial_codes);
        std::free(initial_ids);
    }

    virtual void TearDown() {
        // Deletes all of the versions of the partition created
        for(auto& version_data : partition->partition_versions_) { 
            std::filesystem::path vectors_path = version_data.data_path_; vectors_path.concat(".vectors");
            std::filesystem::remove(vectors_path);

            std::filesystem::path ids_path = version_data.data_path_; ids_path.concat(".idxs");
            std::filesystem::remove(ids_path);
        }
    }

    // Helper function to generate sequential codes
    void generate_sequential_codes(size_t n, std::vector<uint8_t>& codes, unsigned int start_val = 0) {
        codes.resize(n * code_size);
        for (size_t i = 0; i < n * code_size; ++i) {
            codes[i] = static_cast<uint8_t>((start_val + i) % 256);
        }
    }

    // Helper function to generate sequential IDs
    void generate_sequential_ids(size_t n, std::vector<idx_t>& ids, idx_t start_id = 0) {
        ids.resize(n);
        for (size_t i = 0; i < n; ++i) {
            ids[i] = start_id + i;
        }
    }

    // Helper functions for verification
    void verify_ids(const idx_t* actual_ids, const std::vector<idx_t>& expected_ids, size_t start_idx = 0) {
        for (size_t i = 0; i < expected_ids.size(); ++i) {
            ASSERT_EQ(actual_ids[start_idx + i], expected_ids[i]) << "Mismatch at index " << (start_idx + i);
        }
    }

    void verify_codes(const uint8_t* actual_codes, const std::vector<uint8_t>& expected_codes, size_t start_idx = 0) {
        size_t code_bytes = code_size;
        for (size_t i = 0; i < expected_codes.size() / code_bytes; ++i) {
            ASSERT_EQ(std::memcmp(actual_codes + (start_idx + i) * code_bytes,
                                  expected_codes.data() + i * code_bytes,
                                  code_bytes), 0)
                << "Code mismatch at vector " << (start_idx + i);
        }
    }
};


TEST_F(OnDiskArrowIndexPartitionTest, InitialGetTest) {
    // First verify that we are properly able to read the saved codes
    const uint8_t* initial_codes = partition->get_codes();
    ASSERT_NE(initial_codes, nullptr);
    verify_codes(initial_codes, curr_codes_vec_);

    // Now verify that we are able to properly read the saves ids
    const idx_t* initial_ids = partition->get_ids();
    ASSERT_NE(initial_ids, nullptr);
    verify_ids(initial_ids, curr_ids_vec_);
}

// Test to ensure that when we load the latest codes/ids that those results
// get cached to make future accesses quicker 
TEST_F(OnDiskArrowIndexPartitionTest, CopyCacheTest) {
    // First measure the time taken for the first get call
    auto start_time = std::chrono::high_resolution_clock::now();
    const uint8_t* initial_codes = partition->get_codes();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto initial_codes_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Now measure the time it takes if the codes are cached
    start_time = std::chrono::high_resolution_clock::now();
    const uint8_t* cached_codes = partition->get_codes();
    end_time = std::chrono::high_resolution_clock::now();
    auto cached_codes_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    ASSERT_LE(cached_codes_us.count(), initial_codes_us.count()); 

    // Now also do the same for ids
    start_time = std::chrono::high_resolution_clock::now();
    const idx_t* initial_ids = partition->get_ids();
    end_time = std::chrono::high_resolution_clock::now();
    auto initial_ids_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    start_time = std::chrono::high_resolution_clock::now();
    const idx_t* cached_ids = partition->get_ids();
    end_time = std::chrono::high_resolution_clock::now();
    auto cached_ids_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    ASSERT_LE(cached_ids_us.count(), initial_ids_us.count());
}

TEST_F(OnDiskArrowIndexPartitionTest, UpdatePartitionIDTest) {
   // Update the partition id associated with this partition
   size_t new_partition_id = 13;
   partition->set_partition_id(new_partition_id);

   // Ensure that this create a new version that is a copy of the previous version with a different name
   size_t num_version = partition->partition_versions_.size();
   DiskArrowPartitionVersion& new_version = partition->partition_versions_[num_version - 1];
   DiskArrowPartitionVersion& prev_version = partition->partition_versions_[num_version - 2];
   ASSERT_EQ(prev_version.num_vectors_, new_version.num_vectors_);
   ASSERT_LE(prev_version.version_timestamp_, new_version.version_timestamp_);
   std::string new_version_data_path = new_version.data_path_.string();
   ASSERT_NE(new_version_data_path.find(std::to_string(new_partition_id)), std::string::npos);
}

TEST_F(OnDiskArrowIndexPartitionTest, ClearPartitionTest) {
   // First cleart the partition
   partition->clear();
   
   // Verify that the latest version is "empty", marking an cleared partition
   DiskArrowPartitionVersion& cleared_version = partition->partition_versions_[partition->partition_versions_.size() - 1];
   ASSERT_EQ(cleared_version.num_vectors_, 0);

   std::filesystem::path vectors_path = cleared_version.data_path_; vectors_path.concat(".vectors");
   ASSERT_EQ(std::filesystem::file_size(vectors_path), 0); 

   std::filesystem::path ids_path = cleared_version.data_path_; ids_path.concat(".idxs");
   ASSERT_EQ(std::filesystem::file_size(ids_path), 0); 
}

TEST_F(OnDiskArrowIndexPartitionTest, FindIDTest) {
    for(size_t i = 0; i < curr_ids_vec_.size(); i++) { 
        ASSERT_EQ(partition->find_id(curr_ids_vec_[i]), i);
    }
}

TEST_F(OnDiskArrowIndexPartitionTest, AppendTest) {
    // Generate a bunch of new entries and append them
    int64_t num_new_entries = initial_num_vectors;
    std::vector<uint8_t> new_codes;
    std::vector<idx_t> new_ids;
    generate_sequential_codes(num_new_entries, new_codes);
    generate_sequential_ids(num_new_entries, new_ids, initial_num_vectors);
    partition->append(num_new_entries, new_ids.data(), new_codes.data()); 

    // Also update the test copy as well
    curr_codes_vec_.insert(curr_codes_vec_.end(), new_codes.begin(), new_codes.end());
    curr_ids_vec_.insert(curr_ids_vec_.end(), new_ids.begin(), new_ids.end());

    // Now perform gets and verify that we see the appended entries
    int64_t total_vectors = static_cast<int64_t>(curr_ids_vec_.size());
    ASSERT_EQ(total_vectors, partition->get_num_vectors());

    const uint8_t* updated_codes = partition->get_codes();
    ASSERT_NE(updated_codes, nullptr);
    verify_codes(updated_codes, curr_codes_vec_);

    const idx_t* updated_ids = partition->get_ids();
    ASSERT_NE(updated_ids, nullptr);
    verify_ids(updated_ids, curr_ids_vec_);
}

TEST_F(OnDiskArrowIndexPartitionTest, UpdateTest) {
    // Generate the updated entries
    int64_t new_entries_starting_offset = initial_num_vectors * 0.25;
    int64_t num_new_entries = initial_num_vectors * 0.3;

    std::vector<uint8_t> new_codes;
    std::vector<idx_t> new_ids;
    generate_sequential_codes(num_new_entries, new_codes, 56);
    generate_sequential_ids(num_new_entries, new_ids, 4 * initial_num_vectors);
    partition->update(new_entries_starting_offset, num_new_entries, new_ids.data(), new_codes.data()); 

    // Also update the test copy as well
    int64_t codes_starting_offset = new_entries_starting_offset * code_size;
    int64_t num_new_codes = static_cast<int64_t>(new_codes.size());
    for(int64_t i = 0; i < num_new_codes; i++) { 
        curr_codes_vec_[codes_starting_offset + i] = new_codes[i];
    }
    for(int64_t i = 0; i < new_ids.size(); i++) { 
        curr_ids_vec_[new_entries_starting_offset + i] = new_ids[i];
    }

    // Now perform gets and verify that we see the updated entries
    int64_t total_vectors = static_cast<int64_t>(curr_ids_vec_.size());
    ASSERT_EQ(total_vectors, partition->get_num_vectors());

    const uint8_t* updated_codes = partition->get_codes();
    ASSERT_NE(updated_codes, nullptr);
    verify_codes(updated_codes, curr_codes_vec_);

    const idx_t* updated_ids = partition->get_ids();
    ASSERT_NE(updated_ids, nullptr);
    verify_ids(updated_ids, curr_ids_vec_);
}

TEST_F(OnDiskArrowIndexPartitionTest, RemoveTest) {
    // Remove the element at an index
    int64_t index_to_remove = 7;
    partition->remove(index_to_remove);

    // Ensure that the number of elements has decreased and the removed element has been swapped
    // with the last element
    int64_t updated_element_count = initial_num_vectors - 1;
    curr_ids_vec_[index_to_remove] = curr_ids_vec_[updated_element_count];
    curr_ids_vec_.resize(updated_element_count);

    int64_t write_offset = index_to_remove * code_size; int64_t read_offset = updated_element_count * code_size;
    for(int64_t i = 0; i < code_size; i++) { 
        curr_codes_vec_[write_offset + i] = curr_codes_vec_[read_offset + i];
    }
    curr_codes_vec_.resize(read_offset);

    // Now perform gets and verify that we see the updated entries
    int64_t total_vectors = static_cast<int64_t>(curr_ids_vec_.size());
    ASSERT_EQ(total_vectors, partition->get_num_vectors());

    const uint8_t* updated_codes = partition->get_codes();
    ASSERT_NE(updated_codes, nullptr);
    verify_codes(updated_codes, curr_codes_vec_);

    const idx_t* updated_ids = partition->get_ids();
    ASSERT_NE(updated_ids, nullptr);
    verify_ids(updated_ids, curr_ids_vec_);
}