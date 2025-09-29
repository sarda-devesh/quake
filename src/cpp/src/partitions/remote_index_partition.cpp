//
// Created by Devesh on 09/25/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include <partitions/remote_index_partition.h>

#include <stdexcept>

RemoteIndexPartition::RemoteIndexPartition(size_t partition_id, int64_t code_size, std::shared_ptr<PartitionInitializeParams> initialize_parameters)  
    : partition_id_(partition_id), code_size_(code_size), intialize_parameters_(initialize_parameters), num_vectors_(0) { 

    // Get the client that can be used to communicate with the storage node
    global_partition_id_ = intialize_parameters_->global_partition_id_;
    storage_client_ = StorageClientStore::GetStorageClient(intialize_parameters_->storage_node_address_);
    bool result = storage_client_->add_partition(global_partition_id_, code_size);
    std::cout << "[Remote Index] " << global_partition_id_ << ": Initializing partition " << partition_id << " with code size " << code_size << " returning result " << result << std::endl;
    assert(result == true);
}

const uint8_t* RemoteIndexPartition::get_codes() { 
    throw std::runtime_error("get_codes Function not implemented");
    return nullptr;
}

const idx_t* RemoteIndexPartition::get_ids() { 
    throw std::runtime_error("get_ids Function not implemented");
    return nullptr;
}

int64_t RemoteIndexPartition::get_num_vectors() { 
    return num_vectors_;
}

int64_t RemoteIndexPartition::get_code_size() { 
    return code_size_;
}

void RemoteIndexPartition::set_code_size(int64_t code_size) { 
    throw std::runtime_error("set_code_size Function not implemented");
}

void RemoteIndexPartition::set_partition_id(size_t partition_id) { 
    throw std::runtime_error("set_partition_id Function not implemented");
}

void RemoteIndexPartition::append(int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) { 
    bool result = storage_client_->add_vectors(global_partition_id_, n_entry, code_size_/sizeof(float), new_ids, reinterpret_cast<const float*>(new_codes));
    num_vectors_ += n_entry;
    std::cout << "[Remote Index] " << global_partition_id_ <<  ": Add vectors called with n entry of " << n_entry << " updating num vectors to " << num_vectors_ << " returning result of " << result << std::endl;
    assert(result == true);
}

void RemoteIndexPartition::update(int64_t offset, int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) { 
    throw std::runtime_error("Update Function not implemented");
}

void RemoteIndexPartition::remove(int64_t index) { 
    throw std::runtime_error("Remove Function not implemented");
}

void RemoteIndexPartition::resize(int64_t new_capacity) { 
    throw std::runtime_error("Resize Function not implemented");
}

void RemoteIndexPartition::clear() { 
    throw std::runtime_error("Clear Function not implemented");
}

int64_t RemoteIndexPartition::find_id(idx_t id) { 
    throw std::runtime_error("Find id Function not implemented");
}

std::pair<std::vector<float>, std::vector<int64_t>> RemoteIndexPartition::get_top_k(size_t k, 
    int num_queries, const float* query_vectors, MetricType metric) { 
    
   // Make the call to the storage node
   return storage_client_->perform_search(global_partition_id_, k, num_queries, code_size_/sizeof(float), 
    query_vectors, metric);     
}