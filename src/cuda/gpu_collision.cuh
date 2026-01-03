// cuda/gpu_collision.cuh
#pragma once

#include <cuda_runtime.h>

#include "gpu_bvh.cuh"
#include "gpu_broadphase.cuh"
#include "type.h"

namespace gpu {

// GPU-side Contact
struct Contact_d {
  I32 body_index_a;
  I32 body_index_b;
  F32_3 position;
  F32_3 normal;
  F32 depth;
};
// GPU-side Plane
struct Plane_d {
  F32_3 normal;
  F32 offset;
};
// GPU-side Body Transform
struct BodyTransform_d {
  F32_3 position;
  F32_4 orientation;  // Quaternion
};

class CollisionDetector_GPU {
public:
  CollisionDetector_GPU();
  ~CollisionDetector_GPU();
  
  void set_BVH(BVHBuilder_GPU* bvh_builder);
  void detect_body_environment(
    I32 body_idx,
    const BodyTransform_d& transform,
    const Plane_d* planes,
    I32 num_planes,
    Vector<Contact_d>& out_contacts
  );
  void detect_body_body(
    I32 body_a_idx, const BodyTransform_d& transform_a, BVHBuilder_GPU* bvh_a,
    I32 body_b_idx, const BodyTransform_d& transform_b, BVHBuilder_GPU* bvh_b,
    Vector<Contact_d>& out_contacts
  );
  void broadphase_detect(
    const Vector<Vec3>& aabb_mins,
    const Vector<Vec3>& aabb_maxs,
    Vector<CollisionPair>& out_pairs
  );
  
  void free();

private:
  // Device memory for contacts
  Contact_d* d_contacts = nullptr;
  I32* d_contact_count  = nullptr;
  I32 max_contacts      = 65536;
  // Device memory for planes
  Plane_d* d_planes     = nullptr;
  I32 num_planes        = 0;
  
  BVHBuilder_GPU* m_bvh = nullptr;
  std::unique_ptr<BroadphaseGPU> m_broadphase;
};

// ============== Device Helper Functions ==============

__device__ inline F32_3 operator+(F32_3 a, F32_3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline F32_3 operator-(F32_3 a, F32_3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline F32_3 operator*(F32_3 a, F32 s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ inline F32 dot(F32_3 a, F32_3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline F32_3 cross(F32_3 a, F32_3 b) {
  return make_float3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );
}
__device__ inline F32_3 rotate_by_quat(F32_3 v, F32_4 q) {
  F32_3 u = make_float3(q.x, q.y, q.z);
  F32 s = q.w;
  return u * (2.0f * dot(u, v)) + v * (s * s - dot(u, u)) + cross(u, v) * (2.0f * s);
}
__device__ inline bool aabb_intersects_plane(const AABB_d& box, F32_3 normal, F32 offset) {
  F32_3 center = (box.min + box.max) * 0.5f;
  F32_3 extents = (box.max - box.min) * 0.5f;
  F32 r = extents.x * fabsf(normal.x) + extents.y * fabsf(normal.y) + extents.z * fabsf(normal.z);
  F32 s = dot(normal, center) - offset;
  return s <= r;
}

} // namespace gpu