set(QUAKE_TEST_HOME ${CMAKE_CURRENT_BINARY_DIR})

find_package(Torch REQUIRED)

file(GLOB SRCS *.cpp)

ADD_EXECUTABLE(quake_tests ${SRCS})
target_include_directories(quake_tests PUBLIC ${TORCH_INCLUDE_DIRS})

TARGET_LINK_LIBRARIES(quake_tests
        ${PROJECT_NAME}
        ${TORCH_LIBRARIES}
        gtest gtest_main
)

add_test(NAME quake_tests COMMAND quake_tests WORKING_DIRECTORY ${QUAKE_TEST_HOME})