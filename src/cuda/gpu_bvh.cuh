// cuda/gpu_bvh.cuh
#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "type.h"

#ifndef __CUDACC__    // If we are NOT compiling with NVCC (e.g., GCC/Clang), define __clz manually.

#include <limits>
inline int __clz(unsigned int x) { return (x == 0) ? 32 : __builtin_clz(x); }

#endif


namespace gpu {

// GPU-side AABB
struct AABB_d {
  F32_3 min;
  F32_3 max;
};
// GPU-side BVH Node 
struct BVHNode_d {
  AABB_d bounds;
  I32 left;
  I32 right;
  I32 first_prim;
  I32 prim_count;
  
  __host__ __device__ bool is_leaf() const { return prim_count > 0; }
};

// GPU BVH Constructor
class BVHBuilder_GPU {
public:
  BVHBuilder_GPU();
  ~BVHBuilder_GPU();

  void build(const Vector<Vec3>& vertices, const Vector<Trig>& triangles);
  void copy_to_host(Vector<BVHNode_d>& out_nodes, Vector<I32>& out_prim_indices) const;
  void free();
  
  BVHNode_d* device_nodes()  const { return d_nodes; }
  I32* device_prim_indices() const { return d_prim_indices; }
  F32_3* device_vertices()   const { return d_vertices; }
  I32_3* device_triangles()  const { return d_triangles; }
  
  I32 node_count()     const { return m_num_nodes; }
  I32 triangle_count() const { return m_num_triangles; }
  I32 vertex_count()   const { return m_num_vertices; }
  I32 root_index()     const { return 0; }

private:
  // Device memory
  F32_3* d_vertices = nullptr;
  I32_3* d_triangles = nullptr;
  F32_3* d_centroids = nullptr;
  AABB_d* d_prim_bounds = nullptr;
  
  U_I32* d_morton_codes = nullptr;
  I32* d_sorted_indices = nullptr;
  I32* d_prim_indices = nullptr;
  
  BVHNode_d* d_nodes = nullptr;        // Internal nodes (n-1) + leaf references
  I32* d_parent_indices = nullptr;     // Parent pointers for bottom-up AABB
  I32* d_atomic_counters = nullptr;    // For bottom-up traversal
  
  I32 m_num_vertices = 0;
  I32 m_num_triangles = 0;
  I32 m_num_internal_nodes = 0;
  I32 m_num_nodes = 0;
  
  void compute_centroids_bounds();
  void compute_morton_codes();
  void sort_morton_codes();
  void build_radix_tree();
  void compute_node_bounds();
};

// Helper: Compute Morton Code 
__device__ __host__ inline unsigned int expand_bits(unsigned int v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}
__device__ __host__ inline unsigned int compute_morton_3d(float x, float y, float z) {
  x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
  y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
  z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
  unsigned int xx = expand_bits((unsigned int)x);
  unsigned int yy = expand_bits((unsigned int)y);
  unsigned int zz = expand_bits((unsigned int)z);
  return (xx << 2) | (yy << 1) | zz;
}
// Longest common prefix
__device__ inline int delta(const unsigned int* sorted_codes, int n, int i, int j) {
  if (j < 0 || j >= n) return -1;
  unsigned int ki = sorted_codes[i];
  unsigned int kj = sorted_codes[j];
  if (ki == kj) 
    return 32 + __clz(i ^ j);
  return __clz(ki ^ kj);
}

} // namespace gpu