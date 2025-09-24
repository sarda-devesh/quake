#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <iostream>
#include <memory>
#include <string>
 #include <cassert>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"

#include <communication/coordinator_client.h>

ABSL_FLAG(uint32_t, port, 8001, "Port to launch this service on");
ABSL_FLAG(std::string, coordinator_address, "localhost:5051", "Address of the coordinator service");

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);

    // First register the server with the coordinator
    std::string coordinator_address = absl::GetFlag(FLAGS_coordinator_address);
    CoordinatorClient coordinator_client(coordinator_address);
    uint16_t service_port = absl::GetFlag(FLAGS_port);
    bool registered_worker = coordinator_client.register_worker(service_port, false);
    assert(registered_worker && "Failed to register storage node with coordinator");

    // Now launch the grpc service for this storage node
    std::string server_address = absl::StrFormat("0.0.0.0:%d", service_port);
    std::cout << "Starting Storage Node at " << server_address << std::endl;

    return 0;
}