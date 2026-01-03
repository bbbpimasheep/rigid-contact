// cuda/gpu_broadphase.cuh
#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "type.h"

namespace gpu {

struct AABBEndpoint {
  float value;      // endpoint coordinate
  int body_id;      // body ID
  int is_max;       // 0 = min end, 1 = max end
};
struct CollisionPair {
  int body_a;
  int body_b;
};

class BroadphaseGPU {
public:
  BroadphaseGPU();
  ~BroadphaseGPU();
  
  void detect_pairs(
    const Vector<Vec3>& aabb_mins,
    const Vector<Vec3>& aabb_maxs,
    Vector<CollisionPair>& out_pairs
  );
  void free();

private:
  AABBEndpoint* d_endpoints = nullptr;
  int* d_sorted_indices     = nullptr;
  int* d_active_list        = nullptr;
  CollisionPair* d_pairs    = nullptr;
  int* d_pair_count         = nullptr;
  
  float* d_aabb_mins        = nullptr;  // [n * 3]
  float* d_aabb_maxs        = nullptr;  // [n * 3]
  
  int m_capacity  = 0;
  int m_max_pairs = 0;
  
  void ensure_capacity(int num_bodies);
};

} // namespace gpu