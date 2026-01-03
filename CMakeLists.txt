cmake_minimum_required(VERSION 3.17)

project(rigid_sim LANGUAGES CXX CUDA)

# ==========================================
# 1. Standard Settings
# ==========================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# ==========================================
# 2. Find Dependencies
# ==========================================
find_package(CUDAToolkit REQUIRED)

# Find TBB (OneTBB)
# Note: This automatically looks for the system-installed TBBConfig.cmake
find_package(TBB REQUIRED)

# ==========================================
# 3. Source File Definitions
# ==========================================
set(CPP_SOURCES
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

set(CUDA_SOURCES
    src/cuda/gpu_bvh.cu
    src/cuda/gpu_broadphase.cu
    src/cuda/gpu_collision.cu
)

add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

# ==========================================
# 4. Include Paths and Linking (Critical Part)
# ==========================================

# Set include paths
target_include_directories(${PROJECT_NAME} PRIVATE 
    src/include
    src/cuda
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/ext
)

# Set link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    CUDA::cudart
    TBB::tbb
)

# ==========================================
# 5. Compile Options (Resolve CUDA + Eigen conflicts)
# ==========================================
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(${PROJECT_NAME} PRIVATE 
    # Allow Eigen to be used in CUDA device code
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    
    # FIX: Use -Xcudafe to pass suppression flags ONLY to the CUDA frontend.
    # This prevents the "unrecognized command-line option" error in GCC.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=20014>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=20013>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=20015>
)

# ==========================================
# 6. Asset Copying (Optional)
# ==========================================
if(EXISTS "${CMAKE_SOURCE_DIR}/assets")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        ${CMAKE_CURRENT_BINARY_DIR}/assets
        COMMENT "Copying assets..."
    )
endif()