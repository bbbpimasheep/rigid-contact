cmake_minimum_required(VERSION 3.17)

# Project name
project(rigid_sim)

# Set C++ standard to C++17 (required for std::filesystem)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Source files setup
# ------------------------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/bvh.cpp
    src/collision.cpp
    src/integrator.cpp
    src/mesh_lib.cpp
    src/properties.cpp
    src/rigid_body.cpp
    src/scene.cpp
    src/simulation.cpp
    src/tri_mesh.cpp
)

# ------------------------------------------------------------------------------
# Create executable
# ------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})

# ------------------------------------------------------------------------------
# Include directories setup
# ------------------------------------------------------------------------------
 
# 1. Internal header files (src/include)
target_include_directories(${PROJECT_NAME} PRIVATE src/include)

# 2. External libraries (Eigen)
# Based on your directory structure: ext/Eigen/Dense
# To use #include <Eigen/Dense>, we need to include the 'ext' directory
target_include_directories(${PROJECT_NAME} PRIVATE ext)

# 3. TBB (oneTBB)
find_package(TBB REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb)

# ------------------------------------------------------------------------------
# Asset handling
# ------------------------------------------------------------------------------
# Automatically copy the assets folder to the build directory after compilation
# This ensures the program can find assets/bunny.obj at runtime
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets
    ${CMAKE_CURRENT_BINARY_DIR}/assets
    COMMENT "Copying assets to build directory..."
)

# ------------------------------------------------------------------------------
# Platform compatibility (may be needed for older Linux/GCC versions)
# ------------------------------------------------------------------------------
if(UNIX AND NOT APPLE)
    # For some older GCC versions, std::filesystem requires linking against stdc++fs
    find_library(FS_LIB stdc++fs)
    if(FS_LIB)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${FS_LIB})
    endif()
endif()