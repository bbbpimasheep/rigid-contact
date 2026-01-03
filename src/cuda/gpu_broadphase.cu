// cuda/gpu_broadphase.cu
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <cstdio>

#include "gpu_broadphase.cuh"

namespace gpu {

// ============== CUDA Kernels ==============

__global__ void create_endpoints_kernel(
  const float* aabb_mins,  // [n * 3]
  const float* aabb_maxs,  // [n * 3]
  AABBEndpoint* endpoints, // [n * 2]
  int n,
  int axis  // sort by which axis
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  // Min endpoint
  endpoints[idx * 2].value    = aabb_mins[idx * 3 + axis];
  endpoints[idx * 2].body_id  = idx;
  endpoints[idx * 2].is_max   = 0;
  // Max endpoint
  endpoints[idx * 2 + 1].value    = aabb_maxs[idx * 3 + axis];
  endpoints[idx * 2 + 1].body_id  = idx;
  endpoints[idx * 2 + 1].is_max   = 1;
}

struct EndpointComparator {
  __host__ __device__ bool operator()(const AABBEndpoint& a, const AABBEndpoint& b) const {
    if (a.value != b.value) return a.value < b.value;
    return a.is_max < b.is_max; // min comes the first when positions are the same
  }
};

__global__ void sweep_prune_kernel(
  const AABBEndpoint* sorted_endpoints,
  const float* aabb_mins,
  const float* aabb_maxs,
  int num_endpoints,
  int num_bodies,
  CollisionPair* pairs,
  int* pair_count,
  int max_pairs
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_endpoints) return;
  
  AABBEndpoint ep = sorted_endpoints[idx];
  
  if (ep.is_max) return;
  
  int body_a = ep.body_id;
  float min_a_x = aabb_mins[body_a * 3 + 0];
  float min_a_y = aabb_mins[body_a * 3 + 1];
  float min_a_z = aabb_mins[body_a * 3 + 2];
  float max_a_x = aabb_maxs[body_a * 3 + 0];
  float max_a_y = aabb_maxs[body_a * 3 + 1];
  float max_a_z = aabb_maxs[body_a * 3 + 2];
  
  for (int j = idx + 1; j < num_endpoints; ++j) {
    AABBEndpoint other = sorted_endpoints[j];
    
    if (other.body_id == body_a && other.is_max) break;
    if (other.is_max) continue;
    if (other.body_id == body_a) continue;
    
    int body_b = other.body_id;
    
    float min_b_x = aabb_mins[body_b * 3 + 0];
    float min_b_y = aabb_mins[body_b * 3 + 1];
    float min_b_z = aabb_mins[body_b * 3 + 2];
    float max_b_x = aabb_maxs[body_b * 3 + 0];
    float max_b_y = aabb_maxs[body_b * 3 + 1];
    float max_b_z = aabb_maxs[body_b * 3 + 2];
    
    bool overlap_x = (min_a_x <= max_b_x) && (max_a_x >= min_b_x);
    bool overlap_y = (min_a_y <= max_b_y) && (max_a_y >= min_b_y);
    bool overlap_z = (min_a_z <= max_b_z) && (max_a_z >= min_b_z);
    
    if (overlap_x && overlap_y && overlap_z) {
      int pair_idx = atomicAdd(pair_count, 1);
      if (pair_idx < max_pairs) {
        // make sure body_a < body_b
        if (body_a < body_b) {
          pairs[pair_idx].body_a = body_a;
          pairs[pair_idx].body_b = body_b;
        } else {
          pairs[pair_idx].body_a = body_b;
          pairs[pair_idx].body_b = body_a;
        }
      }
    }
  }
}
__global__ void compute_axis_variance_kernel(
  const float* aabb_mins,
  const float* aabb_maxs,
  float* axis_variance,
  int n
) {
  __shared__ float sum[3];
  __shared__ float sum_sq[3];
  
  if (threadIdx.x < 3) {
    sum[threadIdx.x] = 0;
    sum_sq[threadIdx.x] = 0;
  }
  __syncthreads();
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    for (int axis = 0; axis < 3; ++axis) {
      float center = (aabb_mins[idx * 3 + axis] + aabb_maxs[idx * 3 + axis]) * 0.5f;
      atomicAdd(&sum[axis], center);
      atomicAdd(&sum_sq[axis], center * center);
    }
  }
  __syncthreads();
  
  if (threadIdx.x < 3 && blockIdx.x == 0) {
    float mean = sum[threadIdx.x] / n;
    float variance = sum_sq[threadIdx.x] / n - mean * mean;
    axis_variance[threadIdx.x] = variance;
  }
}

// ============== Shared Memory Sweep ==============

#define BLOCK_SIZE 256
#define MAX_ACTIVE_PER_BLOCK 512

__global__ void sweep_optimized_kernel(
  const AABBEndpoint* sorted_endpoints,
  const float* aabb_mins,
  const float* aabb_maxs,
  int num_endpoints,
  CollisionPair* pairs,
  int* pair_count,
  int max_pairs
) {
    __shared__ int active_bodies[MAX_ACTIVE_PER_BLOCK];
  __shared__ int active_count;
  
  int tid = threadIdx.x;
  int block_start = blockIdx.x * BLOCK_SIZE;
  
  if (tid == 0) active_count = 0;
  __syncthreads();
  
  if (tid == 0) {
    for (int i = 0; i < block_start && i < num_endpoints; ++i) {
      AABBEndpoint ep = sorted_endpoints[i];
      if (!ep.is_max) {
        float max_val = aabb_maxs[ep.body_id * 3];  // use x axis
        if (max_val >= sorted_endpoints[block_start].value)
          if (active_count < MAX_ACTIVE_PER_BLOCK) 
              active_bodies[active_count++] = ep.body_id;
      }
    }
  }
  __syncthreads();
  
  int idx = block_start + tid;
  if (idx >= num_endpoints) return;
  
  AABBEndpoint ep = sorted_endpoints[idx];
  
  if (!ep.is_max) {
    int body_a = ep.body_id;
    float min_a[3], max_a[3];
    for (int k = 0; k < 3; ++k) {
      min_a[k] = aabb_mins[body_a * 3 + k];
      max_a[k] = aabb_maxs[body_a * 3 + k];
    }
    
    for (int i = 0; i < active_count; ++i) {
      int body_b = active_bodies[i];
      if (body_b == body_a) continue;
      
      float min_b[3], max_b[3];
      for (int k = 0; k < 3; ++k) {
        min_b[k] = aabb_mins[body_b * 3 + k];
        max_b[k] = aabb_maxs[body_b * 3 + k];
      }
      
      bool overlap = true;
      for (int k = 0; k < 3; ++k) {
        if (min_a[k] > max_b[k] || max_a[k] < min_b[k]) {
          overlap = false;
          break;
        }
      }
      
      if (overlap) {
        int pair_idx = atomicAdd(pair_count, 1);
        if (pair_idx < max_pairs) {
          pairs[pair_idx].body_a = min(body_a, body_b);
          pairs[pair_idx].body_b = max(body_a, body_b);
        }
      }
    }
  }
}

// ============== BroadphaseGPU 实现 ==============

BroadphaseGPU::BroadphaseGPU() { cudaMalloc(&d_pair_count, sizeof(int)); }
BroadphaseGPU::~BroadphaseGPU() { free(); }

void BroadphaseGPU::free() {
  if (d_endpoints)  cudaFree(d_endpoints);
  if (d_aabb_mins)  cudaFree(d_aabb_mins);
  if (d_aabb_maxs)  cudaFree(d_aabb_maxs);
  if (d_pairs)      cudaFree(d_pairs);
  if (d_pair_count) cudaFree(d_pair_count);
  
  d_endpoints   = nullptr;
  d_aabb_mins   = nullptr;
  d_aabb_maxs   = nullptr;
  d_pairs       = nullptr;
  d_pair_count  = nullptr;
  m_capacity    = 0;
}

void BroadphaseGPU::ensure_capacity(int num_bodies) {
  if (num_bodies <= m_capacity) return;
  
  if (d_endpoints)  cudaFree(d_endpoints);
  if (d_aabb_mins)  cudaFree(d_aabb_mins);
  if (d_aabb_maxs)  cudaFree(d_aabb_maxs);
  if (d_pairs)      cudaFree(d_pairs);
  if (d_pair_count) cudaFree(d_pair_count);
  
  m_capacity = num_bodies << 1;
  m_max_pairs = (m_capacity * m_capacity) >> 2;
  if (m_max_pairs > 1e6) m_max_pairs = 1e6;
  
  cudaMalloc(&d_endpoints,  m_capacity  * 2 * sizeof(AABBEndpoint));
  cudaMalloc(&d_aabb_mins,  m_capacity  * 3 * sizeof(float));
  cudaMalloc(&d_aabb_maxs,  m_capacity  * 3 * sizeof(float));
  cudaMalloc(&d_pairs,      m_max_pairs     * sizeof(CollisionPair));
  cudaMalloc(&d_pair_count, sizeof(int));

  cudaMemset(d_pair_count, 0, sizeof(int));
}

void BroadphaseGPU::detect_pairs(
  const Vector<Vec3>& aabb_mins,
  const Vector<Vec3>& aabb_maxs,
  Vector<CollisionPair>& out_pairs
) {
  int n = aabb_mins.size();
  if (n < 2) {
      out_pairs.clear();
      return;
  }
  
  ensure_capacity(n);
  // Copy AABB to GPU
  Vector<float> h_mins(n * 3), h_maxs(n * 3);
  for (int i = 0; i < n; ++i) {
    h_mins[i * 3 + 0] = aabb_mins[i].x();
    h_mins[i * 3 + 1] = aabb_mins[i].y();
    h_mins[i * 3 + 2] = aabb_mins[i].z();
    h_maxs[i * 3 + 0] = aabb_maxs[i].x();
    h_maxs[i * 3 + 1] = aabb_maxs[i].y();
    h_maxs[i * 3 + 2] = aabb_maxs[i].z();
  }
  cudaMemcpy(d_aabb_mins, h_mins.data(), n * 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_aabb_maxs, h_maxs.data(), n * 3 * sizeof(float), cudaMemcpyHostToDevice);
  
// --- 调试检查开始 ---
  size_t addr_ep = reinterpret_cast<uintptr_t>(d_endpoints);
  if (addr_ep % 16 != 0) { // 因为之前是 align(16)，这里严格检查
    printf("[FATAL] Frame ???: d_endpoints is NOT 16-byte aligned! Addr: %zu\n", addr_ep);
    exit(1);
  }
// --- 调试检查结束 ---

  int axis = 0; // choose x
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  
  create_endpoints_kernel<<<numBlocks, blockSize>>>(
    d_aabb_mins, d_aabb_maxs, d_endpoints, n, axis
  );
  
  // thrust::device_ptr<AABBEndpoint> ep_ptr(d_endpoints);
  try {
    // thrust::sort(ep_ptr, ep_ptr + n * 2, EndpointComparator());
    thrust::sort(thrust::device, d_endpoints, d_endpoints + n * 2, EndpointComparator());
  } catch (thrust::system_error &e) {
    // printf("[CRASH] Thrust sort failed inside Broadphase: %s\n", e.what());
    printf("[FATAL] Thrust sort failed: %s\n", e.what());
    exit(1);
  }
  
  int zero = 0;
  cudaMemcpy(d_pair_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
  
  int num_endpoints = n * 2;
  numBlocks = (num_endpoints + blockSize - 1) / blockSize;
  
  sweep_prune_kernel<<<numBlocks, blockSize>>>(
    d_endpoints,
    d_aabb_mins,
    d_aabb_maxs,
    num_endpoints,
    n,
    d_pairs,
    d_pair_count,
    m_max_pairs
  );
  
  cudaDeviceSynchronize();
  
  int pair_count;
  cudaMemcpy(&pair_count, d_pair_count, sizeof(int), cudaMemcpyDeviceToHost);
  pair_count = min(pair_count, m_max_pairs);
  
  out_pairs.resize(pair_count);
  if (pair_count > 0)
    cudaMemcpy(out_pairs.data(), d_pairs, pair_count * sizeof(CollisionPair), cudaMemcpyDeviceToHost);
  
  std::sort(out_pairs.begin(), out_pairs.end(), [](const CollisionPair& a, const CollisionPair& b) {
    if (a.body_a != b.body_a) return a.body_a < b.body_a;
    return a.body_b < b.body_b;
  });
  auto last = std::unique(out_pairs.begin(), out_pairs.end(), [](const CollisionPair& a, const CollisionPair& b) {
    return a.body_a == b.body_a && a.body_b == b.body_b;
  });
  out_pairs.erase(last, out_pairs.end());
}

} // namespace gpu