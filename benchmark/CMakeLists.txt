cmake_minimum_required(VERSION 3.10)
project(TestICP_PCL)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find PCL
find_package(PCL REQUIRED)

# Include PCL headers
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add the executable
add_executable(speed_test_comparison speed_test_comparison.cpp)

# Link PCL libraries
target_link_libraries(speed_test_comparison ${PCL_LIBRARIES})
