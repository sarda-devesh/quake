//
// Created by Devesh on 08/05/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef COORDINATOR_CLIENT_H
#define COORDINATOR_CLIENT_H

#include <grpcpp/grpcpp.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cassert>
#include <common.h>

#include "coordinator.grpc.pb.h" 

using coordinator::Coordinator;

/**
 * @brief Wrapper to interact with the Coordinator service via gRPC.
 *
 * The CoordinatorClient is a client-side wrapper that facilitates communication
 * with the Coordinator service using gRPC. All communication to the Coordinator
 * should be done through this class.
 */
class CoordinatorClient { 
public: 
    static std::shared_ptr<CoordinatorClient> GetCoordinatorClient() { 
        assert(client_instance != nullptr);
        return client_instance;
    }

    static void SetGetCoordinatorClient(std::shared_ptr<CoordinatorClient> coordinator_client) { 
        client_instance = coordinator_client;
    }

    /** 
    * @brief Constructor for CoordinatorClient
    * @param coordinator_address The address of the coordinator service (e.g., "localhost:5051").
    */
    CoordinatorClient(std::string coordinator_address);

    /** 
     * @brief Method to register the current worker with the coordinator
     * 
     * @param service_port The port where the caller is hosting the grpc service that can be used to communicate with it.
     * @param is_compute_worker A boolean indicating if the worker is a compute worker (true) or storage worker (false)
     * @return True if registration was successful; false otherwise.
    */
    bool register_worker(uint32_t service_port, bool is_compute_worker);


    /**
     * @brief Method to register a new index with the coordinator
     * 
     * @param num_partitions The number of partitions in the new index
     * @return A struct representing all of the details related to the new index
     */
    std::shared_ptr<DistributedIndexDetails> register_new_index(int num_partitions);

private:
    inline static std::shared_ptr<CoordinatorClient> client_instance = nullptr;
    
    std::unique_ptr<Coordinator::Stub> stub_; ///< gRPC stub for making calls to the Coordinator service.
};



#endif //COORDINATOR_CLIENT_H