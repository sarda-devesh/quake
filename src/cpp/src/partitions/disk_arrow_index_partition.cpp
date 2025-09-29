//
// Created by Jason on 12/18/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include <partitions/disk_arrow_index_partition.h>

OnDiskArrowIndexPartition::OnDiskArrowIndexPartition(size_t partition_id, 
                               int64_t num_vectors,
                               uint8_t* codes,
                               idx_t* ids,
                               int64_t code_size) 
    : ids_copy_(nullptr), codes_copy_(nullptr), code_size_(code_size), partition_id_(partition_id) 
{
    add_new_partition_version(codes, ids, num_vectors);
}

OnDiskArrowIndexPartition::OnDiskArrowIndexPartition(size_t partition_id, int64_t code_size)
    : ids_copy_(nullptr), codes_copy_(nullptr), code_size_(code_size), partition_id_(partition_id) 
{ 

}

OnDiskArrowIndexPartition::~OnDiskArrowIndexPartition() {
    // Free the allocated buffers
    if(ids_copy_ != nullptr) { 
        std::free(ids_copy_);
    }

    if(codes_copy_ != nullptr) { 
        std::free(codes_copy_);
    }
}

const uint8_t* OnDiskArrowIndexPartition::get_codes() {
    // See if the copy already exists
    size_t read_version =  partition_versions_.size() - 1;
    DiskArrowPartitionVersion& read_version_details = partition_versions_[read_version];
    if(codes_copy_version_ == read_version && codes_copy_ != nullptr) { 
       return codes_copy_;
    }

    // Now see if we need to allocate a new buffer or realloc an existing buffer
    uint64_t updated_buffer_capacity = codes_copy_capacity_;
    uint64_t read_buffer_size = read_version_details.num_vectors_ * code_size_ * sizeof(uint8_t); 
    if(codes_copy_ == nullptr) { 
        codes_copy_ = reinterpret_cast<uint8_t*>(std::malloc(read_buffer_size));
        updated_buffer_capacity = read_buffer_size;
    } else if(read_buffer_size > codes_copy_capacity_) { 
        codes_copy_ = reinterpret_cast<uint8_t*>(std::realloc(codes_copy_, read_buffer_size));
        updated_buffer_capacity = read_buffer_size;
    }
    codes_copy_capacity_ = updated_buffer_capacity;

    // Now load the data from disk
    std::filesystem::path vectors_path = read_version_details.data_path_; vectors_path.concat(".vectors");
    read_buffer_from_disk(vectors_path, codes_copy_, codes_copy_capacity_);

    // Return a pointer to the updated buffer
    codes_copy_version_ = read_version; 
    return codes_copy_;
}

const idx_t* OnDiskArrowIndexPartition::get_ids() { 
    // See if the copy already exists
    size_t read_version =  partition_versions_.size() - 1;
    DiskArrowPartitionVersion& read_version_details = partition_versions_[read_version];
    if(ids_copy_version_ == read_version && ids_copy_ != nullptr) { 
       return ids_copy_;
    }

    // Now see if we need to allocate a new buffer or realloc an existing buffer
    uint64_t updated_buffer_capacity = ids_copy_capacity_;
    uint64_t read_buffer_size = read_version_details.num_vectors_ * sizeof(idx_t); 
    if(ids_copy_ == nullptr) { 
        ids_copy_ = reinterpret_cast<idx_t*>(std::malloc(read_buffer_size));
        updated_buffer_capacity = read_buffer_size;
    } else if(read_buffer_size > ids_copy_capacity_) { 
        ids_copy_ = reinterpret_cast<idx_t*>(std::realloc(ids_copy_, read_buffer_size));
        updated_buffer_capacity = read_buffer_size;
    }
    ids_copy_capacity_ = updated_buffer_capacity;

    // Now load the data from disk
    std::filesystem::path ids_path = read_version_details.data_path_; ids_path.concat(".idxs");
    read_buffer_from_disk(ids_path, reinterpret_cast<uint8_t*>(ids_copy_), ids_copy_capacity_);

    // Return a pointer to the updated buffer
    ids_copy_version_ = read_version; 
    return ids_copy_;
}

int64_t OnDiskArrowIndexPartition::get_num_vectors() {
    return partition_versions_[partition_versions_.size() - 1].num_vectors_;
}

void OnDiskArrowIndexPartition::set_partition_id(size_t partition_id) { 
    // Update the id associated with this partition 
    partition_id_ = partition_id;
    if(partition_versions_.empty()) return;

    // Also determine the paths for the new versions file
    auto current_time = std::chrono::system_clock::now();
    int64_t new_version_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time.time_since_epoch()).count();
    std::stringstream file_name_stream;
    file_name_stream << "partition_" << partition_id_ << "_" << new_version_timestamp;
    std::filesystem::path new_common_save_path = file_name_stream.str();
    std::filesystem::path new_vector_save_path = new_common_save_path; new_vector_save_path.concat(".vectors");
    std::filesystem::path new_ids_save_path = new_common_save_path; new_ids_save_path.concat(".idxs");

    // Now create the new version by copying the files from the old version as the data says the same
    DiskArrowPartitionVersion& latest_version_details = partition_versions_[partition_versions_.size() - 1];
    std::filesystem::path src_vectors_path = latest_version_details.data_path_; src_vectors_path.concat(".vectors");
    std::filesystem::copy(src_vectors_path, new_vector_save_path, std::filesystem::copy_options::overwrite_existing);

    std::filesystem::path src_ids_path = latest_version_details.data_path_; src_ids_path.concat(".idxs");
    std::filesystem::copy(src_ids_path, new_ids_save_path, std::filesystem::copy_options::overwrite_existing);

    // Finally save the new version
    size_t num_existing_versions = partition_versions_.size();
    DiskArrowPartitionVersion new_parition_version = {
        static_cast<int64_t>(num_existing_versions),
        new_version_timestamp,
        latest_version_details.num_vectors_,
        new_common_save_path 
    };
    partition_versions_.push_back(new_parition_version);
}

void OnDiskArrowIndexPartition::append(int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) { 
    if(n_entry == 0) {
        return;
    }

    std::cout << "Append called with num entry of " << n_entry << " with code size of " << code_size_ << std::endl;

    // Deal with the case that we don't have any existing partition
    if(partition_versions_.empty()) { 
        std::cout << "OnDiskArrowIndexPartition: Append called with no existing partition so just making it the new partition " << std::endl;
        add_new_partition_version(new_codes, new_ids, n_entry);
        return;
    }

    // Allocate buffer for the new vectors
    DiskArrowPartitionVersion& latest_version_details = partition_versions_[partition_versions_.size() - 1];
    int64_t num_curr_vectors = latest_version_details.num_vectors_;
    int64_t num_new_vectors = num_curr_vectors + n_entry; 
    uint8_t* codes_ptr = reinterpret_cast<uint8_t*>(std::malloc(num_new_vectors * code_size_));
    idx_t* ids_ptr = reinterpret_cast<idx_t*>(std::malloc(num_new_vectors * sizeof(idx_t)));

    // Read the existing vectors and ids from disk
    std::filesystem::path vectors_path = latest_version_details.data_path_; vectors_path.concat(".vectors");
    read_buffer_from_disk(vectors_path, codes_ptr, num_curr_vectors * code_size_);

    std::filesystem::path ids_path = latest_version_details.data_path_; ids_path.concat(".idxs");
    read_buffer_from_disk(ids_path, reinterpret_cast<uint8_t*>(ids_ptr), num_curr_vectors * sizeof(idx_t));

    // Now append the values and write back the updated valeus to disk
    std::memcpy(codes_ptr + num_curr_vectors * code_size_, new_codes, n_entry * code_size_);
    std::memcpy(ids_ptr + num_curr_vectors, new_ids, n_entry * sizeof(idx_t));
    add_new_partition_version(codes_ptr, ids_ptr, num_new_vectors);

    // Free the allocated buffer because we only allocate the buffer for this method
    std::free(codes_ptr); std:;free(ids_ptr);
}

void OnDiskArrowIndexPartition::update(int64_t offset, int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) { 
    // Validate input arguments
    if (n_entry <= 0) {
        throw std::runtime_error("n_entry must be positive in update");
    }

    DiskArrowPartitionVersion& latest_version_details = partition_versions_[partition_versions_.size() - 1];
    int64_t num_curr_vectors = latest_version_details.num_vectors_;
    if (offset < 0 || offset + n_entry > num_curr_vectors) {
        throw std::runtime_error("Offset + n_entry out of range in update");
    }

    // Load the existing vectors
    uint8_t* codes_ptr = reinterpret_cast<uint8_t*>(std::malloc(num_curr_vectors * code_size_));
    idx_t* ids_ptr = reinterpret_cast<idx_t*>(std::malloc(num_curr_vectors * sizeof(idx_t)));

    // Read the existing vectors and ids from disk
    std::filesystem::path vectors_path = latest_version_details.data_path_; vectors_path.concat(".vectors");
    read_buffer_from_disk(vectors_path, codes_ptr, num_curr_vectors * code_size_);

    std::filesystem::path ids_path = latest_version_details.data_path_; ids_path.concat(".idxs");
    read_buffer_from_disk(ids_path, reinterpret_cast<uint8_t*>(ids_ptr), num_curr_vectors * sizeof(idx_t));

    // Update the vectors from the specified offsets and save them as a new version
    std::memcpy(codes_ptr + offset * code_size_, new_codes, n_entry * code_size_);
    std::memcpy(ids_ptr + offset, new_ids, n_entry * sizeof(idx_t));
    add_new_partition_version(codes_ptr, ids_ptr, num_curr_vectors);

    // Free the allocated buffer because we only allocate the buffer for this method
    std::free(codes_ptr); std:;free(ids_ptr);
}

void OnDiskArrowIndexPartition::remove(int64_t index) { 
    DiskArrowPartitionVersion& latest_version_details = partition_versions_[partition_versions_.size() - 1];
    int64_t num_curr_vectors = latest_version_details.num_vectors_;
    if (index < 0 || index >= num_curr_vectors) {
        throw std::runtime_error("Index out of range in remove");
    }

    // Load the existing vectors
    uint8_t* codes_ptr = reinterpret_cast<uint8_t*>(std::malloc(num_curr_vectors * code_size_));
    idx_t* ids_ptr = reinterpret_cast<idx_t*>(std::malloc(num_curr_vectors * sizeof(idx_t)));

    // Read the existing vectors and ids from disk
    std::filesystem::path vectors_path = latest_version_details.data_path_; vectors_path.concat(".vectors");
    read_buffer_from_disk(vectors_path, codes_ptr, num_curr_vectors * code_size_);

    std::filesystem::path ids_path = latest_version_details.data_path_; ids_path.concat(".idxs");
    read_buffer_from_disk(ids_path, reinterpret_cast<uint8_t*>(ids_ptr), num_curr_vectors * sizeof(idx_t));

    // Swap the element at the specified index with the element at the last index and save this as a new version
    int64_t last_element_idx = num_curr_vectors - 1;
    std::memcpy(codes_ptr + index * code_size_, codes_ptr + last_element_idx * code_size_, code_size_);
    ids_ptr[index] = ids_ptr[last_element_idx];
    add_new_partition_version(codes_ptr, ids_ptr, last_element_idx);

    // Free the allocated buffer because we only allocate the buffer for this method
    std::free(codes_ptr); std:;free(ids_ptr);
}

void OnDiskArrowIndexPartition::resize(int64_t new_capacity) { 
    // This is a no op as we don't allocate any buffers in memory

}

void OnDiskArrowIndexPartition::clear() { 
    // Clear by making a new version that is emtpy
    add_new_partition_version(nullptr, nullptr, 0);
}

int64_t OnDiskArrowIndexPartition::find_id(idx_t id) { 
    int64_t num_ids = get_num_vectors();
    const idx_t* ids_copy_ptr = get_ids();
    for(int64_t i = 0; i < num_ids; i++) { 
        if(ids_copy_ptr[i] == id) { 
            return i;
        }
    }

    return -1;
}

void OnDiskArrowIndexPartition::add_new_partition_version(const uint8_t* codes, const idx_t* ids, int64_t num_vectors) {
    auto current_time = std::chrono::system_clock::now();
    int64_t version_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time.time_since_epoch()).count();
    
    // Determine the prefix of the save path
    int64_t num_existing_versions = static_cast<int64_t>(partition_versions_.size());
    std::stringstream file_name_stream;
    file_name_stream << "partition_" << partition_id_ << "_" << num_existing_versions << "_" << version_timestamp;
    std::filesystem::path common_save_path = file_name_stream.str();

    // Now write the vectors and ids to disk
    std::filesystem::path vector_save_path = common_save_path; vector_save_path.concat(".vectors");
    std::filesystem::path ids_save_path = common_save_path; ids_save_path.concat(".idxs");

    // Now persist the data for this version to disk
    if(num_vectors > 0) { 
        std::cout << "add_new_partition_version writing " << num_vectors << " vectors with code size of " << code_size_ << " to path " << vector_save_path << std::endl;
        write_buffer_to_disk(codes, num_vectors * code_size_, vector_save_path); 
        std::cout << "add_new_partition_version writing " << num_vectors << " ids to path " << ids_save_path << std::endl;
        write_buffer_to_disk(reinterpret_cast<const uint8_t*>(ids), num_vectors * sizeof(idx_t), ids_save_path); 
    } else { 
        std::ofstream code_file(vector_save_path);
        std::ofstream ids_file(ids_save_path);
    }

    // Finally save this partition
    DiskArrowPartitionVersion new_parition_version = {
        num_existing_versions,
        version_timestamp,
        num_vectors,
        common_save_path 
    };
    partition_versions_.push_back(new_parition_version);
    std::cout << "Updated num partitions to " << num_vectors << std::endl;
}

void OnDiskArrowIndexPartition::write_buffer_to_disk(const uint8_t* buffer, int64_t buffer_size, std::filesystem::path save_path) { 
    std::ofstream outFile(save_path, std::ios::out | std::ios::binary);
    assert(outFile.is_open());
    outFile.write(reinterpret_cast<const char*>(buffer), buffer_size);
    outFile.close();
}

int64_t OnDiskArrowIndexPartition::read_buffer_from_disk(std::filesystem::path data_path, uint8_t* buffer, int64_t buffer_size) { 
    std::ifstream inFile(data_path, std::ios::in | std::ios::binary);
    assert(inFile.is_open());
    inFile.read(reinterpret_cast<char*>(buffer), buffer_size);
    int64_t bytes_read = inFile.gcount();
    inFile.close();
    return bytes_read;
}