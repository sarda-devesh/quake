//
// Created by Devesh on 09/26/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include <partitions/partition_store.h>
#include <partitions/disk_arrow_index_partition.h>
#include <partitions/in_memory_index_partition.h>

PartitionStore::~PartitionStore() { 
    shutdown_workers();   
}

void PartitionStore::add_partition(size_t partition_id, int64_t code_size, bool store_partition_on_disk) { 
    if(partitions_.find(partition_id) != partitions_.end()) { 
        throw std::runtime_error("Partition already in partition store");
    }

    if(store_partition_on_disk) { 
        partitions_[partition_id] = std::make_shared<OnDiskArrowIndexPartition>(partition_id, code_size);
    } else { 
        partitions_[partition_id] = std::make_shared<InMemoryIndexPartition>(0, nullptr, nullptr, code_size);
        partitions_[partition_id]->set_partition_id(partition_id);
    }
    
    if constexpr(debug_) std::cout << "Partition Store: Created partition " << partition_id << " with code size of " << code_size << " with on disk of " << store_partition_on_disk << std::endl;
}

void PartitionStore::add_vectors(size_t partition_id, int64_t n_entry, const idx_t* ids, const float* codes) { 
    if(partitions_.find(partition_id) == partitions_.end()) { 
        throw std::runtime_error("Partition not in partition store");
    }

    if constexpr(debug_) std::cout << "Partition Store: For partition " << partition_id << " adding " << n_entry << " entries" << std::endl;
    partitions_[partition_id]->append(n_entry, ids, reinterpret_cast<const uint8_t*>(codes));
}

int64_t PartitionStore::get_dimension(size_t partition_id)  { 
    if(partitions_.find(partition_id) == partitions_.end()) { 
        throw std::runtime_error("Partition not in partition store");
    }

    return partitions_[partition_id]->get_code_size()/sizeof(float);
}

void PartitionStore::initialize_workers(int num_workers) { 
    if(!worker_threads_.empty()) { 
        throw std::runtime_error("Workers are already running");
    }

    worker_threads_.resize(num_workers);
    worker_resources_.resize(num_workers);
    for(int i = 0; i < num_workers; i++) { 
        worker_threads_[i] = std::thread(&PartitionStore::partition_search_worker_fn, this, i);
    }
    std::cout << "Initialized the partition store with " << num_workers << " workers" << std::endl;
}

void PartitionStore::shutdown_workers() { 
    // Submit the shutdown signal to each worker
    for(size_t i = 0; i < worker_threads_.size(); i++) { 
        PartitionSearchJob shutdown_job;
        shutdown_job.partition_id = -1;
        job_queue_.enqueue(shutdown_job);
    }

    // Wait for the threads to shutdown
    for (auto &thread : worker_threads_) {
        if (thread.joinable()) { 
            thread.join();
        }
    }
    worker_threads_.clear();
}

PartitionSearchResult PartitionStore::perform_search(size_t partition_id, size_t k, int num_queries, const float* query_vectors, 
    MetricType metric, int64_t* result_ids, float* result_distances) { 
    
    if(partitions_.find(partition_id) == partitions_.end()) { 
        throw std::runtime_error("Partition not in partition store");
    }

    // Create the search job and submit it
    PartitionSearchResult search_result;

    auto job_create_start = std::chrono::high_resolution_clock::now();
    std::promise<void>* promise_ptr = new std::promise<void>;
    std::future<void> future = promise_ptr->get_future();

    PartitionSearchJob search_job;
    search_job.partition_id = static_cast<int64_t>(partition_id);
    search_job.k = k;
    search_job.query_vector = query_vectors;
    search_job.num_queries = num_queries;
    search_job.result_ids = result_ids;
    search_job.result_distances = result_distances;
    search_job.promise = promise_ptr;
    search_job.metric_type = metric;
    auto job_create_end = std::chrono::high_resolution_clock::now();
    search_result.job_init_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(job_create_end - job_create_start).count();

    auto job_enque_start = std::chrono::high_resolution_clock::now();
    job_queue_.enqueue(search_job);
    if constexpr(debug_) std::cout << "Submit search job to the job queue" << std::endl;
    auto job_enque_end = std::chrono::high_resolution_clock::now();
    search_result.job_enqueue_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(job_enque_end - job_enque_start).count();
    
    // Wait for the search to be finished
    auto job_wait_start = std::chrono::high_resolution_clock::now();
    future.get();
    if constexpr(debug_) std::cout << "Got the worker finished the search" << std::endl;
    delete promise_ptr;
    auto job_wait_end = std::chrono::high_resolution_clock::now();
    search_result.job_wait_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(job_wait_end - job_wait_start).count();

    return search_result;
}

inline void PartitionStore::record_worker_metrics(WorkerResources& res, std::string metric_name, float metric_value) {
    if(res.metrics.find(metric_name) == res.metrics.end()) { 
        res.metrics[metric_name] = std::make_shared<MetricStore>(metric_name);
    }
    res.metrics[metric_name]->add_value(metric_value);
}

void PartitionStore::partition_search_worker_fn(int worker_id) {
    // Pin the worker to a core
    int core_id = (worker_id + THREAD_CORE_MAPPING_OFFSET) % std::thread::hardware_concurrency();
    assert(set_thread_affinity(core_id));
    
    // Initialize the thread resources
    WorkerResources& res = worker_resources_[worker_id];

    std::vector<std::shared_ptr<TopkBuffer>> topk_buffers;
    topk_buffers.resize(DEFAULT_TOP_K_BUFFER_SIZE);
    for(int i = 0; i < DEFAULT_TOP_K_BUFFER_SIZE; i++) { 
        topk_buffers[i] = std::make_shared<TopkBuffer>(1, false);
    }

    while (true) {
        // Try to get a search job
        PartitionSearchJob search_job;
        job_queue_.wait_dequeue(search_job);

        // See if it is a shutdown signal
        if(search_job.partition_id == -1) { 
            break;
        }

        // Make sure we have enough local buffers
        auto search_initialize_start = std::chrono::high_resolution_clock::now();
        int num_queries = search_job.num_queries;
        while(topk_buffers.size() < num_queries) { 
            topk_buffers.push_back(std::make_shared<TopkBuffer>(1, false));
        }

        // Retrieve the partition
        std::shared_ptr<IndexPartition> partition = partitions_[search_job.partition_id];
        const float* partition_vecs = reinterpret_cast<const float*>(partition->get_codes());
        const int64_t* partition_ids = reinterpret_cast<const int64_t*>(partition->get_ids());
        int num_vectors = partition->get_num_vectors();
        int dimension = partition->get_code_size()/sizeof(float);
        size_t results_per_query = search_job.k;
        if constexpr(debug_) std::cout << "Perform search on partition " << search_job.partition_id << " with " << num_vectors << " vectors with top k of " << results_per_query << " for " << num_queries << " queries" << std::endl;
        auto search_initialize_end = std::chrono::high_resolution_clock::now();

        int64_t search_initialize_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(search_initialize_end - search_initialize_start).count();
        record_worker_metrics(res, "search_initialize_ms", search_initialize_ns/MS_TO_NS);
        
        if(num_queries == 1) { 
            // Perform a normal search
            auto buffer_reset_start = std::chrono::high_resolution_clock::now();
            std::shared_ptr<TopkBuffer> query_buffer = topk_buffers[0];
            query_buffer->set_k(results_per_query);
            query_buffer->is_descending_ = search_job.metric_type == faiss::METRIC_INNER_PRODUCT;
            query_buffer->reset();
            auto buffer_reset_end = std::chrono::high_resolution_clock::now();
            int64_t buffer_reset_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(buffer_reset_end - buffer_reset_start).count();
            record_worker_metrics(res, "search_buffer_reset_ms", buffer_reset_ns/MS_TO_NS);

            auto scan_start = std::chrono::high_resolution_clock::now();
            scan_list(search_job.query_vector, partition_vecs, partition_ids, num_vectors, dimension, *query_buffer, search_job.metric_type);
            auto scan_end = std::chrono::high_resolution_clock::now();
            int64_t scan_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(scan_end - scan_start).count();
            record_worker_metrics(res, "search_scan_list_ms", scan_ns/MS_TO_NS);

            // Write out the output result
            if constexpr(debug_) std::cout << "Single Query Results: ";
            auto result_write_start = std::chrono::high_resolution_clock::now();
            std::vector<float> topk_distances = query_buffer->get_topk();
            std::vector<int64_t> topk_indicies = query_buffer->get_topk_indices();
            for(size_t i = 0; i < topk_distances.size(); i++) { 
                search_job.result_distances[i] = topk_distances[i];
                search_job.result_ids[i] = topk_indicies[i];
                if constexpr(debug_) std::cout << i << "/" << results_per_query << " - (" << topk_indicies[i] << "," << topk_distances[i] << "); ";
            }
            if constexpr(debug_) std::cout << std::endl;

            for(size_t i = topk_distances.size(); i < results_per_query; i++) { 
                search_job.result_ids[i] = -1;
            }
            auto result_write_end = std::chrono::high_resolution_clock::now();
            int64_t result_write_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(result_write_end - result_write_start).count();
            record_worker_metrics(res, "search_result_write_ms", result_write_ns/MS_TO_NS);
        } else { 
            // Perform a batched search
            topk_buffers.resize(num_queries);
            for(int i = 0; i < num_queries; i++) { 
                topk_buffers[i]->set_k(results_per_query);
                topk_buffers[i]->is_descending_ = search_job.metric_type == faiss::METRIC_INNER_PRODUCT;
                topk_buffers[i]->reset();
            }

            batched_scan_list(search_job.query_vector, partition_vecs, partition_ids, num_queries, num_vectors, dimension, topk_buffers, search_job.metric_type);

            // Write out the output result for each query
            for(int i = 0; i < num_queries; i++) { 
                std::vector<float> topk_distances = topk_buffers[i]->get_topk();
                std::vector<int64_t> topk_indicies = topk_buffers[i]->get_topk_indices();

                float* result_dist_write_ptr = search_job.result_distances + i * results_per_query;
                int64_t* result_ids_write_ptr = search_job.result_ids + i * results_per_query;
                for(size_t j = 0; j < topk_distances.size(); j++) { 
                    result_dist_write_ptr[i] = topk_distances[i];
                    result_ids_write_ptr[i] = topk_indicies[i];
                }

                for(size_t j = topk_distances.size(); j < results_per_query; j++) { 
                    result_ids_write_ptr[i] = -1;
                }
            }
        }

        // Mark that the job was completed
        if constexpr(debug_) std::cout << "Marking that the worker completed the search " << std::endl;
        search_job.promise->set_value();
    }
}