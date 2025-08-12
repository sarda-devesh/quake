// disk_arrow_index_partition.cpp

#include <gtest/gtest.h>
#include "disk_arrow_index_partition.h"  // Include the OnDiskArrowIndexPartition header
#include <vector>
#include <cstring>

using namespace faiss;

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
        // Initialize initial_ids with sequential IDs starting from 1000
        generate_sequential_ids(initial_num_vectors, curr_ids_vec_, 1000);

        // Initialize initial_codes with sequential codes starting from 0
        generate_sequential_codes(initial_num_vectors, curr_codes_vec_, 0);

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
            EXPECT_EQ(actual_ids[start_idx + i], expected_ids[i]) << "Mismatch at index " << (start_idx + i);
        }
    }

    void verify_codes(const uint8_t* actual_codes, const std::vector<uint8_t>& expected_codes, size_t start_idx = 0) {
        size_t code_bytes = code_size;
        for (size_t i = 0; i < expected_codes.size() / code_bytes; ++i) {
            EXPECT_EQ(std::memcmp(actual_codes + (start_idx + i) * code_bytes,
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