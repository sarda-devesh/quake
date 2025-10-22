//
// Created by Devesh on 09/26/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include <common.h>
#include <partitions/index_partition.h>
#include <list_scanning.h>
#include <blockingconcurrentqueue.h>
#include <parallel.h>

#include <unordered_map>
#include <memory>
#include <thread>
#include <queue>
#include <future>
#include <functional>
#include <vector>
#include <mutex>
#include <condition_variable>

constexpr int DEFAULT_NUM_PARTITION_STORE_WORKERS = 4;
constexpr int DEFAULT_TOP_K_BUFFER_SIZE = 10;
constexpr int THREAD_CORE_MAPPING_OFFSET = 3; // Offset used to map thread to core 

/**
 * @brief Structure representing a search job.
 *
 * A PartitionSearchJob encapsulates all parameters required to scan an index partition for matching vectors
 */
struct PartitionSearchJob {
 int64_t partition_id;         ///< The identifier of the partition to be scanned.
 size_t k;                     ///< The number of neighbors (Top-K) to return.
 const float* query_vector;    ///< Pointer to the query vector(s).
 int num_queries;              ///< The number of queries in batched mode.
 int64_t* result_ids;          ///< The pointer to the buffer to write the result ids
 float* result_distances;      ///< The pointer to the buffer to write the result distances
 std::promise<void>* promise;  ///< Promise that can be used by the worker to indicate that search is complete
 MetricType metric_type;       ///< The metric to use for the distance calculation
};

/**
 * @brief Struct containing the result of a partition search
 */
struct PartitionSearchResult { 
    int64_t job_init_time_ns; // The time taken to inititalize the scan job
    int64_t job_enqueue_time_ns; // Time taken to add the job to the queue
    int64_t job_wait_time_ns; // The time spent waiting for a result
};

/**
 * @brief Class used by the Storage Node in order to manage multiple partitions and to perform operating on them
 * 
 * Responsibilities:
 *  - Add/Update/Remove vectors from the approriate partitions
 *  - Perform search operations on different parititions using a thread pool
*/
class PartitionStore { 
public: 
    /**
     * @brief Struct storing worker specific resources
     */
    struct WorkerResources { 
        std::unordered_map<std::string, std::shared_ptr<MetricStore>> metrics; ///< Map storing worker specific metrics
    };

    std::vector<WorkerResources> worker_resources_; // Vector storing the resources for each worker

    /**
     * @brief Constructor for PartitionStore.
     */
    PartitionStore() = default;

    /**
     * @brief Destructor.
     */
    ~PartitionStore();

    /**
     * @brief Add a new empty partition into the partition store
     * 
     * @param partition_id The id of the partition to add
     * @param code_size The code size of the vectors in this partition
     * @param store_partition_on_disk Whether we should store the partition on disk or in memory
     * 
     * @throws std::runtime_error if this partition already exists
     */
    void add_partition(size_t partition_id, int64_t code_size, bool store_partition_on_disk = true);

    /**
     * @brief Gets the dimension associated with the vectors in the specified partition
     * 
     * @param partition_id The id of the partition to add
     * 
     * @throws std::runtime_error if this partition already exists
     */
    int64_t get_dimension(size_t partition_id);

    /**
     * @brief Append new entries (codes and IDs) to an existing partition.
     *
     * @param partition_id The id of the partition
     * @param n_entry Number of entries to add.
     * @param ids Pointer to the vector IDs.
     * @param vectors Pointer to the encoded vectors.
     *
     * @throws std::runtime_error if the partition does not exist.
     */
    void add_vectors(
        size_t partition_id,
        int64_t n_entry,
        const idx_t* ids,
        const float* vectors);

    /**
     * @brief Perform search on the specified partition for the nearest vector 
     * 
     * @param partition_id The id of the partition
     * @param k The number of neighbors to return
     * @param num_queries The number of query vector
     * @param query_vectors A pointer to the query vector
     * @param metric The type of the metric for the distance calculation
     * @param result_ids The pointer to write the result ids
     * @param result_distances The pointer to write the result distances
     * 
     * @throws std::runtime_error if the partition does not exist.
     */
    PartitionSearchResult perform_search(size_t partition_id, size_t k, int num_queries, const float* query_vectors, 
        MetricType metric, int64_t* result_ids, float* result_distances);     

     /**
     * @brief Initializes worker threads for parallel search of partitions.
     *
     * Spawns worker threads that actually perform the partition scanning job
     *
     * @param num_workers Number of worker threads to initialize.
     * 
     * @throws std::runtime_error if there are workers that are currently running
     */
    void initialize_workers(int num_workers = DEFAULT_NUM_PARTITION_STORE_WORKERS);   

    /**
     * @brief Shuts down all worker threads.
     *
     * Signals each worker to terminate and waits for their completion.
     */
    void shutdown_workers();

private:
    /**
     * @brief Function executed by each worker thread.
     *
     * Processes search process from the job queue
     *
     * @param worker_id The id of the current worker
     */
    void partition_search_worker_fn(int worker_id);  
    
    void record_worker_metrics(WorkerResources& res, std::string metric_name, float metric_value);

    std::vector<std::thread> worker_threads_;  ///< Container for worker threads.
    moodycamel::BlockingConcurrentQueue<PartitionSearchJob> job_queue_; // Queue of the current scan jobs
    std::unordered_map<size_t, std::shared_ptr<IndexPartition>> partitions_; // A map storing all of the partitions
    static constexpr bool debug_ = false; ///< If true, print debug information.
};