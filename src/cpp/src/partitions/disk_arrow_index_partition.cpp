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
    : ids_copy_(nullptr), codes_copy_(nullptr)
                                {
    code_size_ = code_size;
    partition_id_ = partition_id;
    add_new_partition_version(codes, ids, num_vectors);
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
    std::stringstream file_name_stream;
    file_name_stream << "partition_" << partition_id_ << "_" << version_timestamp;
    std::filesystem::path common_save_path = file_name_stream.str();

    // Now write the vectors and ids to disk
    std::filesystem::path vector_save_path = common_save_path; vector_save_path.concat(".vectors");
    std::filesystem::path ids_save_path = common_save_path; ids_save_path.concat(".idxs");

    // Now persist the data for this version to disk
    if(num_vectors > 0) { 
        write_buffer_to_disk(codes, num_vectors * code_size_ * sizeof(uint8_t), vector_save_path); 
        write_buffer_to_disk(reinterpret_cast<const uint8_t*>(ids), num_vectors * sizeof(idx_t), ids_save_path); 
    } else { 
        std::ofstream code_file(vector_save_path);
        std::ofstream ids_file(ids_save_path);
    }

    // Finally save this partition
    size_t num_existing_versions = partition_versions_.size();
    DiskArrowPartitionVersion new_parition_version = {
        static_cast<int64_t>(num_existing_versions),
        version_timestamp,
        num_vectors,
        common_save_path 
    };
    partition_versions_.push_back(new_parition_version);
}

void OnDiskArrowIndexPartition::write_buffer_to_disk(const uint8_t* buffer, int64_t buffer_size, std::filesystem::path save_path) { 
    // Build the array from the buffer
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    arrow::UInt8Builder buffer_builder(pool);
    assert(buffer_builder.AppendValues(buffer, buffer_size).ok());
    std::shared_ptr<arrow::Array> buffer_arr;
    assert(buffer_builder.Finish(&buffer_arr).ok());

    // Create the table from the array
    auto col_field = arrow::field(std::string("values"), buffer_arr->type());
    auto table_schema = arrow::schema({col_field});
    auto table_to_save = arrow::Table::Make(table_schema, {buffer_arr});

    // Create the writer that we can use to write the file
    arrow::Result<shared_ptr<arrow::io::FileOutputStream>> output_file_result = arrow::io::FileOutputStream::Open(save_path);
    assert(output_file_result.ok());
    auto output_file = output_file_result.ValueOrDie();

    arrow::Result<shared_ptr<arrow::ipc::RecordBatchWriter>> ipc_writer_result = arrow::ipc::MakeFileWriter(output_file, table_schema);
    assert(ipc_writer_result.ok());
    shared_ptr<arrow::ipc::RecordBatchWriter> ipc_writer = ipc_writer_result.ValueOrDie();

    // Now actually write the table
    assert(ipc_writer->WriteTable(*table_to_save).ok());
    assert(ipc_writer->Close().ok());
}

int64_t OnDiskArrowIndexPartition::read_buffer_from_disk(std::filesystem::path data_path, uint8_t* buffer, int64_t buffer_size) { 
    // Create the reader to read this file
    arrow::Result<shared_ptr<arrow::io::ReadableFile>> data_file_result = arrow::io::ReadableFile::Open(data_path);
    assert(data_file_result.ok());
    auto data_file = data_file_result.ValueOrDie();

    arrow::Result<shared_ptr<arrow::ipc::RecordBatchFileReader>> record_reader_result = arrow::ipc::RecordBatchFileReader::Open(data_file);
    assert(record_reader_result.ok());
    auto record_reader = record_reader_result.ValueOrDie();

    // Read the table from the input file
    arrow::Result<shared_ptr<arrow::Table>> loaded_table_result = record_reader->ToTable();
    assert(loaded_table_result.ok());
    shared_ptr<arrow::Table> loaded_table = loaded_table_result.ValueOrDie();
    shared_ptr<arrow::ChunkedArray> values_column = loaded_table->column(0);

    // Read the values into the output buffer a chunk at a time
    int64_t bytes_read = 0; int num_chunks = values_column->num_chunks();
    for(int chunk_idx = 0; chunk_idx < num_chunks && bytes_read < buffer_size; chunk_idx++) { 
        // Reinterpret this chunk as an uint8_t buffer
        auto curr_chunk_result = values_column->chunk(chunk_idx)->View(arrow::uint8());
        assert(curr_chunk_result.ok());
        auto curr_chunk = std::static_pointer_cast<arrow::UInt8Array>(curr_chunk_result.ValueOrDie());

        // Get the raw data and copy the over the values
        int64_t chunk_bytes = curr_chunk->length();
        int64_t num_bytes_to_copy = std::min(buffer_size - bytes_read, chunk_bytes);
        std::memcpy(buffer + bytes_read, curr_chunk->raw_values(), num_bytes_to_copy);
        bytes_read += num_bytes_to_copy;
    }

    return bytes_read;
}