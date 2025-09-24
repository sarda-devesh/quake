cmake_minimum_required(VERSION 3.24.2)
project(quake_c VERSION 0.1 LANGUAGES CXX)

# ---------------------------------------------------------------
# Important System Variables and Their Defaults
# ---------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(Torch_USE_CUDA OFF CACHE BOOL "Force disable CUDA in Torch")
set(Torch_NO_CUDA ON CACHE BOOL "Force disable CUDA in Torch")
set(USE_CUDA OFF CACHE BOOL "Force disable CUDA globally")

# QUAKE_ENABLE_GPU: Enable GPU support for Faiss
# Default: OFF
if(QUAKE_ENABLE_GPU)
    set(FAISS_ENABLE_GPU ON)
else()
    set(FAISS_ENABLE_GPU OFF)
endif()

if(QUAKE_ENABLE_GPU)
    add_compile_definitions(FAISS_ENABLE_GPU)
endif()

if(QUAKE_USE_NUMA)
    add_compile_definitions(QUAKE_USE_NUMA)
endif()

if(ARROW_LINK_STATIC)
    add_compile_definitions(ARROW_LINK_STATIC)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
if(NOT DEFINED QUAKE_SET_ABI_MODE)
    set(QUAKE_SET_ABI_MODE ON)
endif()

# If in a conda environment, favor conda packages
if(EXISTS $ENV{CONDA_PREFIX})
    message(STATUS "Conda environment detected. Using conda packages.")
    set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})
endif()

# ---------------------------------------------------------------
# Find Torch Path via Python
# ---------------------------------------------------------------
execute_process(
        COMMAND python3 -c "import torch, os; print(os.path.dirname(torch.__file__), end='')"
        OUTPUT_VARIABLE TorchPath
)
set(CMAKE_PREFIX_PATH ${TorchPath} ${CMAKE_PREFIX_PATH})

# ---------------------------------------------------------------
# Project Directories
# ---------------------------------------------------------------
set(CPP_SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/cpp)
set(project_INCLUDE_DIR ${CPP_SOURCE}/include)
set(project_SOURCE_DIR ${CPP_SOURCE}/src)
set(project_BINDINGS_DIR ${CPP_SOURCE}/bindings)
set(project_THIRD_PARTY_DIR ${CPP_SOURCE}/third_party)
set(project_TEST_DIR test/cpp)

# ---------------------------------------------------------------
# Print out Compiler and Path Information
# ---------------------------------------------------------------
message(STATUS "--------- Build Configuration Summary ---------")
message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Project Version: ${PROJECT_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "QUAKE_ENABLE_GPU: ${QUAKE_ENABLE_GPU} (FAISS_ENABLE_GPU: ${FAISS_ENABLE_GPU})")
message(STATUS "QUAKE_USE_NUMA: ${QUAKE_USE_NUMA}")

# Apple-specific adjustments
if(APPLE)
    include_directories("/opt/homebrew/opt/openblas/include")
    set(FAISS_ENABLE_GPU OFF)
endif()

# Compiler options and definitions
add_compile_options(-march=native)

# Switch ABI mode
if(QUAKE_SET_ABI_MODE)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()


# ---------------------------------------------------------------
# Find Required Packages
# ---------------------------------------------------------------
find_package(Arrow REQUIRED)
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development Interpreter REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf REQUIRED)
find_package(absl REQUIRED)

message(STATUS "Torch include dir: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Python include dir: ${Python3_INCLUDE_DIRS}")

set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})

add_subdirectory(${project_THIRD_PARTY_DIR})

# ---------------------------------------------------------------
# Protobuf Dependencies
# ---------------------------------------------------------------

# Set variables for gRPC and Protobuf
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_GRPC_GRPCPP gRPC::grpc++)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

# Proto files directory
set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/protos")
file(GLOB PROTO_FILES "${PROTO_DIR}/*.proto")

# Variables for generated sources and headers
set(PROTO_SRCS "")
set(PROTO_HDRS "")
set(GRPC_SRCS "")
set(GRPC_HDRS "")

# Loop over all proto files and generate commands
foreach(proto_file ${PROTO_FILES})
  get_filename_component(proto_name ${proto_file} NAME_WE)
  get_filename_component(proto_file_path ${proto_file} PATH)

  set(proto_src "${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.pb.cc")
  set(proto_hdr "${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.pb.h")
  set(grpc_src "${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.grpc.pb.cc")
  set(grpc_hdr "${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.grpc.pb.h")

  add_custom_command(
    OUTPUT ${proto_src} ${proto_hdr} ${grpc_src} ${grpc_hdr}
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${proto_file_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${proto_file}"
    DEPENDS ${proto_file}
  )

  list(APPEND PROTO_SRCS ${proto_src})
  list(APPEND PROTO_HDRS ${proto_hdr})
  list(APPEND GRPC_SRCS ${grpc_src})
  list(APPEND GRPC_HDRS ${grpc_hdr})
  message(STATUS "Adding dependencies for proto file: ${proto_file}")
endforeach()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_library(my_proto_lib
  ${PROTO_SRCS} ${PROTO_HDRS}
  ${GRPC_SRCS} ${GRPC_HDRS}
)

target_link_libraries(my_proto_lib
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
)

target_include_directories(my_proto_lib PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Protobuf_INCLUDE_DIRS}
  ${gRPC_INCLUDE_DIRS}
)

# Needed to avoid protobuf undefined symbol issue.
# See https://stackoverflow.com/questions/73047153/protobuf-ld-undefined-symbol-internalmetadata
# for details
target_compile_definitions(my_proto_lib PUBLIC -DNDEBUG)

# ---------------------------------------------------------------
# Sources and Library
# ---------------------------------------------------------------
file(GLOB_RECURSE project_SOURCES ${project_SOURCE_DIR}/*.cpp)
add_library(${PROJECT_NAME} SHARED ${project_SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${TORCH_INCLUDE_DIRS}
        ${project_INCLUDE_DIR}
        ${project_THIRD_PARTY_DIR}/concurrentqueue/
        faiss
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# ---------------------------------------------------------------
# Set Up Link Libraries Conditionally
# ---------------------------------------------------------------
# Base libraries common to all platforms
set(LINK_LIBS
        ${TORCH_LIBRARIES}
        -ffast-math
        -lpthread
        -fPIC
)

if(APPLE)
    list(APPEND LINK_LIBS faiss)
elseif(UNIX)
    # Linux or other UNIX-likes
    if (QUAKE_USE_AVX512)
        list(APPEND LINK_LIBS faiss_avx512)
    else()
        list(APPEND LINK_LIBS faiss)
    endif()

    if(QUAKE_USE_NUMA)
        list(APPEND LINK_LIBS -lnuma)
    endif()

    list(APPEND LINK_LIBS my_proto_lib absl::flags absl::flags_parse absl::strings)
else()
    # unsupported platform
    message(FATAL_ERROR "Unsupported platform")
endif()

if(ARROW_LINK_STATIC)
    list(APPEND LINK_LIBS Arrow::arrow_static)
else()
    list(APPEND LINK_LIBS Arrow::arrow_shared)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC ${LINK_LIBS})

IF(CMAKE_BUILD_TYPE MATCHES Debug AND QUAKE_USE_TSAN)
    message("Using thread sanitizer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set(CMAKE_MODULE_LINKER_FLAGS "{$CMAKE_MODULE_LINKER_FLAGS} -fsanitize=thread")
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug AND QUAKE_USE_TSAN)

IF(CMAKE_BUILD_TYPE MATCHES Debug AND QUAKE_USE_ASAN)
    message("Using address sanitizer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-recover=address")
    set(CMAKE_MODULE_LINKER_FLAGS "{$CMAKE_MODULE_LINKER_FLAGS} -fsanitize=address")
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug AND QUAKE_USE_ASAN)

# ---------------------------------------------------------------
# Python Bindings
# ---------------------------------------------------------------
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
file(GLOB_RECURSE BINDINGS ${project_BINDINGS_DIR}/*.cpp)
pybind11_add_module(_bindings ${BINDINGS})
set_target_properties(_bindings PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(_bindings PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})
add_custom_target(bindings)
add_dependencies(bindings _bindings)

# ---------------------------------------------------------------
# Leviathan
# ---------------------------------------------------------------
file(GLOB LEVIATHAN_SRCS "src/cpp/leaviathan/*.cpp")

foreach(SOURCE ${LEVIATHAN_SRCS})
    get_filename_component(EXECUTABLE_NAME ${SOURCE} NAME_WE)
    add_executable(${EXECUTABLE_NAME} ${SOURCE})
    target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${LINK_LIBS} ${PROJECT_NAME})
    target_include_directories(${EXECUTABLE_NAME}
            PUBLIC
            ${TORCH_INCLUDE_DIRS}
            ${project_INCLUDE_DIR}
            faiss
    )
    target_compile_features(${EXECUTABLE_NAME} PRIVATE cxx_std_17)
    message(STATUS "Building executable for worker: ${EXECUTABLE_NAME}")
endforeach()

# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(${project_TEST_DIR})
endif()

# ---------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------
message(STATUS "--------- Final Configuration Summary ---------")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "GPU Enabled: ${FAISS_ENABLE_GPU}")
message(STATUS "NUMA Enabled: ${QUAKE_USE_NUMA}")
message(STATUS "Python used: ${Python3_EXECUTABLE}")
message(STATUS "Torch Path: ${TorchPath}")
message(STATUS "MKL_ARCH: ${MKL_ARCH}")
message(STATUS "MKL_ROOT: ${MKL_ROOT}")
message(STATUS "MKL_LINK: ${MKL_LINK}")
message(STATUS "MKL_INTERFACE_FULL: ${MKL_INTERFACE_FULL}")
message(STATUS "MKL_THREADING: ${MKL_THREADING}")
message(STATUS "MKL_MPI: ${MKL_MPI}")
message(STATUS "------------------------------------------------")