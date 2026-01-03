// cuda/gpu_bvh.cu
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <cstdio>

#include "gpu_bvh.cuh"

namespace gpu {

// ============== CUDA Kernels ==============

__global__ void compute_centroids_bounds_kernel(
  const F32_3* vertices,
  const I32_3* triangles,
  F32_3* centroids,
  AABB_d* prim_bounds,
  F32_3* scene_min,
  F32_3* scene_max,
  int n
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  
  I32_3 tri = triangles[idx];
  F32_3 v0 = vertices[tri.x];
  F32_3 v1 = vertices[tri.y];
  F32_3 v2 = vertices[tri.z];

  centroids[idx] = make_float3(
      (v0.x + v1.x + v2.x) / 3.0f,
      (v0.y + v1.y + v2.y) / 3.0f,
      (v0.z + v1.z + v2.z) / 3.0f
  );
  
  AABB_d bounds;
  bounds.min = make_float3(
      fminf(fminf(v0.x, v1.x), v2.x),
      fminf(fminf(v0.y, v1.y), v2.y),
      fminf(fminf(v0.z, v1.z), v2.z)
  );
  bounds.max = make_float3(
      fmaxf(fmaxf(v0.x, v1.x), v2.x),
      fmaxf(fmaxf(v0.y, v1.y), v2.y),
      fmaxf(fmaxf(v0.z, v1.z), v2.z)
  );
  prim_bounds[idx] = bounds;
  
  // Update AABB
  atomicMin((int*)&scene_min->x, __float_as_int(bounds.min.x));
  atomicMin((int*)&scene_min->y, __float_as_int(bounds.min.y));
  atomicMin((int*)&scene_min->z, __float_as_int(bounds.min.z));
  atomicMax((int*)&scene_max->x, __float_as_int(bounds.max.x));
  atomicMax((int*)&scene_max->y, __float_as_int(bounds.max.y));
  atomicMax((int*)&scene_max->z, __float_as_int(bounds.max.z));
}
__global__ void compute_morton_codes_kernel(
  const F32_3* centroids,
  U_I32* morton_codes,
  F32_3 scene_min,
  F32_3 scene_extent,
  int n
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  
  F32_3 c = centroids[idx];
  
  float nx = (scene_extent.x > 0) ? (c.x - scene_min.x) / scene_extent.x : 0.5f;
  float ny = (scene_extent.y > 0) ? (c.y - scene_min.y) / scene_extent.y : 0.5f;
  float nz = (scene_extent.z > 0) ? (c.z - scene_min.z) / scene_extent.z : 0.5f;
  
  morton_codes[idx] = compute_morton_3d(nx, ny, nz);
}
__global__ void build_radix_tree_kernel(
  const U_I32* sorted_morton_codes,
  const int* sorted_indices,
  BVHNode_d* nodes,
  int* parent_indices,
  int* prim_indices,
  int n  // number of leaves
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n - 1) return;

  int d_left  = delta(sorted_morton_codes, n, i, i - 1);
  int d_right = delta(sorted_morton_codes, n, i, i + 1);
  int d = (d_right > d_left) ? 1 : -1;
  
  int delta_min = delta(sorted_morton_codes, n, i, i - d);
  int l_max = 2;
  while (delta(sorted_morton_codes, n, i, i + l_max * d) > delta_min)
    l_max *= 2;
  
  // Binary search for the other side
  int l = 0;
  for (int t = l_max / 2; t >= 1; t /= 2)
    if (delta(sorted_morton_codes, n, i, i + (l + t) * d) > delta_min)
        l = l + t;
  int j = i + l * d;
  
  // Find split gamma
  int delta_node = delta(sorted_morton_codes, n, i, j);
  int s = 0;
  int div = 2;
  int t = (l + div - 1) / div;  // ceil division
  while (t >= 1) {
      if (delta(sorted_morton_codes, n, i, i + (s + t) * d) > delta_node)
          s = s + t;
      div *= 2;
      t = (l + div - 1) / div;
  }
  int γαμμα = i + s * d + min(d, 0);
  
  int left_idx, right_idx;
  int range_left  = min(i, j);
  int range_right = max(i, j);
  if (range_left == γαμμα) {  // left is a leaf
    left_idx = n - 1 + γαμμα;
    nodes[left_idx].first_prim = γαμμα;
    nodes[left_idx].prim_count = 1;
    nodes[left_idx].left = -1;
    nodes[left_idx].right = -1;
    prim_indices[γαμμα] = sorted_indices[γαμμα];
    parent_indices[left_idx] = i;
  } else {
    left_idx = γαμμα;
    parent_indices[γαμμα] = i;
  }
  
  if (range_right == γαμμα + 1) { // right is a leaf
    right_idx = n - 1 + γαμμα + 1;
    nodes[right_idx].first_prim = γαμμα + 1;
    nodes[right_idx].prim_count = 1;
    nodes[right_idx].left = -1;
    nodes[right_idx].right = -1;
    prim_indices[γαμμα + 1] = sorted_indices[γαμμα + 1];
    parent_indices[right_idx] = i;
  } else {
      right_idx = γαμμα + 1;
      parent_indices[γαμμα + 1] = i;
  }
  
  // Internal nodes
  nodes[i].left  = left_idx;
  nodes[i].right = right_idx;
  nodes[i].first_prim = -1;
  nodes[i].prim_count = 0;
}
__global__ void compute_node_bounds_kernel(
  BVHNode_d* nodes,
  const int* parent_indices,
  int* atomic_counters,
  const AABB_d* prim_bounds,
  const int* prim_indices,
  int n
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  
  int leaf_idx = n - 1 + idx;
  int prim_idx = prim_indices[idx];
  nodes[leaf_idx].bounds = prim_bounds[prim_idx];
  int current = parent_indices[leaf_idx];
  while (current >= 0) {
    int old = atomicAdd(&atomic_counters[current], 1);
    if (old == 0) // First arrived thread quits，waiting for peers
        return;
    // Second arrived
    int left  = nodes[current].left;
    int right = nodes[current].right;
    AABB_d left_bounds  = nodes[left].bounds;
    AABB_d right_bounds = nodes[right].bounds;
    
    nodes[current].bounds.min = make_float3(
      fminf(left_bounds.min.x, right_bounds.min.x),
      fminf(left_bounds.min.y, right_bounds.min.y),
      fminf(left_bounds.min.z, right_bounds.min.z)
    );
    nodes[current].bounds.max = make_float3(
      fmaxf(left_bounds.max.x, right_bounds.max.x),
      fmaxf(left_bounds.max.y, right_bounds.max.y),
      fmaxf(left_bounds.max.z, right_bounds.max.z)
    );
    
    current = parent_indices[current];
  }
}

// ============== BVHBuilder_GPU Implementation ==============

BVHBuilder_GPU::BVHBuilder_GPU() {}
BVHBuilder_GPU::~BVHBuilder_GPU() { free(); }

void BVHBuilder_GPU::free() {
  if (d_vertices)         cudaFree(d_vertices);
  if (d_triangles)        cudaFree(d_triangles);
  if (d_centroids)        cudaFree(d_centroids);
  if (d_prim_bounds)      cudaFree(d_prim_bounds);
  if (d_morton_codes)     cudaFree(d_morton_codes);
  if (d_sorted_indices)   cudaFree(d_sorted_indices);
  if (d_prim_indices)     cudaFree(d_prim_indices);
  if (d_nodes)            cudaFree(d_nodes);
  if (d_parent_indices)   cudaFree(d_parent_indices);
  if (d_atomic_counters)  cudaFree(d_atomic_counters);
  
  d_vertices        = nullptr;
  d_triangles       = nullptr;
  d_centroids       = nullptr;
  d_prim_bounds     = nullptr;
  d_morton_codes    = nullptr;
  d_sorted_indices  = nullptr;
  d_prim_indices    = nullptr;
  d_nodes           = nullptr;
  d_parent_indices  = nullptr;
  d_atomic_counters = nullptr;
}
void BVHBuilder_GPU::build(const Vector<Vec3>& vertices, const Vector<Trig>& triangles) {
  free();
  
  m_num_vertices = vertices.size();
  m_num_triangles = triangles.size();
  m_num_internal_nodes = m_num_triangles - 1;
  m_num_nodes = m_num_triangles + m_num_internal_nodes;
  
  if (m_num_triangles == 0) return;
  
  cudaMalloc(&d_vertices,         m_num_vertices        * sizeof(F32_3));
  cudaMalloc(&d_triangles,        m_num_triangles       * sizeof(I32_3));
  cudaMalloc(&d_centroids,        m_num_triangles       * sizeof(F32_3));
  cudaMalloc(&d_prim_bounds,      m_num_triangles       * sizeof(AABB_d));
  cudaMalloc(&d_morton_codes,     m_num_triangles       * sizeof(U_I32));
  cudaMalloc(&d_sorted_indices,   m_num_triangles       * sizeof(int));
  cudaMalloc(&d_prim_indices,     m_num_triangles       * sizeof(int));
  cudaMalloc(&d_nodes,            m_num_nodes           * sizeof(BVHNode_d));
  cudaMalloc(&d_parent_indices,   m_num_nodes           * sizeof(int));
  cudaMalloc(&d_atomic_counters,  m_num_internal_nodes  * sizeof(int));
  
  // Copy verts and trigs to GPU
  std::vector<F32_3> h_vertices(m_num_vertices);
  for (int i = 0; i < m_num_vertices; ++i) 
      h_vertices[i] = make_float3(vertices[i].x(), vertices[i].y(), vertices[i].z());
  cudaMemcpy(d_vertices, h_vertices.data(), m_num_vertices * sizeof(F32_3), cudaMemcpyHostToDevice);
  
  std::vector<I32_3> h_triangles(m_num_triangles);
  for (int i = 0; i < m_num_triangles; ++i)
      h_triangles[i] = make_int3(triangles[i].x(), triangles[i].y(), triangles[i].z());
  cudaMemcpy(d_triangles, h_triangles.data(), m_num_triangles * sizeof(I32_3), cudaMemcpyHostToDevice);
  
  cudaMemset(d_atomic_counters, 0, m_num_internal_nodes * sizeof(int));
  cudaMemset(d_parent_indices, -1, m_num_nodes * sizeof(int));  // root parent = -1
  
  compute_centroids_bounds();
  compute_morton_codes();
  sort_morton_codes();
  build_radix_tree();
  compute_node_bounds();
  
  cudaDeviceSynchronize();
}
void BVHBuilder_GPU::compute_centroids_bounds() {
  F32_3* d_scene_min;
  F32_3* d_scene_max;
  cudaMalloc(&d_scene_min, sizeof(F32_3));
  cudaMalloc(&d_scene_max, sizeof(F32_3));
  
  F32_3 init_min = make_float3( 1e30f,  1e30f,  1e30f);
  F32_3 init_max = make_float3(-1e30f, -1e30f, -1e30f);
  cudaMemcpy(d_scene_min, &init_min, sizeof(F32_3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scene_max, &init_max, sizeof(F32_3), cudaMemcpyHostToDevice);
  
  int blockSize = 256;
  int numBlocks = (m_num_triangles + blockSize - 1) / blockSize;
  
  compute_centroids_bounds_kernel<<<numBlocks, blockSize>>>(
    d_vertices, d_triangles, d_centroids, d_prim_bounds,
    d_scene_min, d_scene_max, m_num_triangles
  );
  
  cudaFree(d_scene_min);
  cudaFree(d_scene_max);
}
void BVHBuilder_GPU::compute_morton_codes() {
    // acquire scene AABB
  thrust::device_ptr<F32_3> centroids_ptr(d_centroids);
  
  std::vector<F32_3> h_centroids(m_num_triangles);
  cudaMemcpy(h_centroids.data(), d_centroids, m_num_triangles * sizeof(F32_3), cudaMemcpyDeviceToHost);
  
  F32_3 scene_min = make_float3( 1e30f,  1e30f,  1e30f);
  F32_3 scene_max = make_float3(-1e30f, -1e30f, -1e30f);
  for (const auto& c : h_centroids) {
    scene_min.x = fminf(scene_min.x, c.x);
    scene_min.y = fminf(scene_min.y, c.y);
    scene_min.z = fminf(scene_min.z, c.z);
    scene_max.x = fmaxf(scene_max.x, c.x);
    scene_max.y = fmaxf(scene_max.y, c.y);
    scene_max.z = fmaxf(scene_max.z, c.z);
  }
  F32_3 scene_extent = make_float3(
    scene_max.x - scene_min.x,
    scene_max.y - scene_min.y,
    scene_max.z - scene_min.z
  );
  
  int blockSize = 256;
  int numBlocks = (m_num_triangles + blockSize - 1) / blockSize;
  
  compute_morton_codes_kernel<<<numBlocks, blockSize>>>(
    d_centroids, d_morton_codes, scene_min, scene_extent, m_num_triangles
  );
}
void BVHBuilder_GPU::sort_morton_codes() {
#if defined RIGID_DEBUGGER
  size_t addr_morton = reinterpret_cast<uintptr_t>(d_morton_codes);
  size_t addr_indices = reinterpret_cast<uintptr_t>(d_sorted_indices);
  printf(stderr, "d_morton_codes address: %zu (Alignment %% 4 = %zu)\n", addr_morton, addr_morton % 4);
  printf(stderr, "d_sorted_indices address: %zu (Alignment %% 4 = %zu)\n", addr_indices, addr_indices % 4);

  if (addr_morton % 4 != 0 || addr_indices % 4 != 0) {
    printf("ERROR: Pointers are NOT aligned to 4 bytes! Crash imminent.\n");
    return; 
  }
#endif
  thrust::device_ptr<U_I32> morton_ptr(d_morton_codes);
  thrust::device_ptr<int> indices_ptr(d_sorted_indices);
  thrust::sequence(indices_ptr, indices_ptr + m_num_triangles);
  thrust::sort_by_key(morton_ptr, morton_ptr + m_num_triangles, indices_ptr);
}
void BVHBuilder_GPU::build_radix_tree() {
  int blockSize = 256;
  int numBlocks = (m_num_internal_nodes + blockSize - 1) / blockSize;
  
  build_radix_tree_kernel<<<numBlocks, blockSize>>>(
    d_morton_codes, d_sorted_indices, d_nodes, d_parent_indices,
    d_prim_indices, m_num_triangles
  );
}
void BVHBuilder_GPU::compute_node_bounds() {
  int blockSize = 256;
  int numBlocks = (m_num_triangles + blockSize - 1) / blockSize;
  
  compute_node_bounds_kernel<<<numBlocks, blockSize>>>(
    d_nodes, d_parent_indices, d_atomic_counters,
    d_prim_bounds, d_prim_indices, m_num_triangles
  );
}
void BVHBuilder_GPU::copy_to_host(Vector<BVHNode_d>& out_nodes, Vector<I32>& out_prim_indices) const {
  out_nodes.resize(m_num_nodes);
  out_prim_indices.resize(m_num_triangles);
  cudaMemcpy(out_nodes.data(),        d_nodes, m_num_nodes * sizeof(BVHNode_d),       cudaMemcpyDeviceToHost);
  cudaMemcpy(out_prim_indices.data(), d_prim_indices, m_num_triangles * sizeof(I32),  cudaMemcpyDeviceToHost);
}

} // namespace gpu