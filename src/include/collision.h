// include/collision.h
#pragma once

#include <vector>
#include <memory>

#include "type.h"
#include "scene.h"
#include "bvh.h"

namespace gpu { 
  class CollisionDetector_GPU; 
  struct Contact_d;
  struct CollisionPair;
}


struct Contact {
  Vec3 position;      // World space contact point
  Vec3 normal;        // World space normal (pointing out of the obstacle)
  F32  depth;         // Penetration depth
  I32  body_index_a;  // Index of the body being penetrated (the obstacle)
  I32  body_index_b;  // Index of the body penetrating (the vertex source), -1 if environment
};

class CollisionSolver {
public:
  CollisionSolver();
  ~CollisionSolver();

  void clear();
  void detect_collisions(Scene& scene);
  const Vector<Contact>& contacts() const;

  void set_use_GPU(bool use) { m_use_gpu = use; }
  bool use_GPU() const { return m_use_gpu; }

private:
  // Checks a single body against the environment boundaries
  void check_be(I32 body_idx, RigidBody& body, Environment& env);
  // Checks collision between two rigid bodies
  void check_bb(I32 idx_a, RigidBody& body_a, I32 idx_b, RigidBody& body_b);
  // Helper: Checks vertices of 'dynamic_body' against 'static_body' mesh
  void check_vm(
    I32 idx_dynamic, RigidBody& dynamic_body,
    I32 idx_static,  RigidBody& static_body
  );
  // Recursive BVH traversal against a plane (Environment)
  void traverse_bvh_plane(
    const BVHNode& node,
    const BVH& bvh,
    const Plane& plane_local,
    const Vector<Vec3>& vertices,
    const Vector<Trig>& triangles,
    const Mat3& body_rot,
    const Vec3& body_pos,
    I32 body_idx,
    const Vec3& plane_world_normal,
    Vector<bool>& visited_verts
  );
  // Recursive BVH traversal for point query (Body-Body)
  // Finds the closest triangle point and checks penetration
  void traverse_bvh_point(
    const BVHNode& node,
    const BVH& bvh,
    const Vec3& point_local,
    const Vector<Vec3>& vertices,
    const Vector<Trig>& triangles,
    F32&  min_dist_sq,
    Vec3& closest_normal,
    Vec3& closest_pos,
    bool& found
  );
  
  void detect_collisions_gpu(Scene& scene);
  void convert_gpu_contacts(const Vector<gpu::Contact_d>& gpu_contacts);
  void broadphase_gpu(Scene& scene, Vector<std::pair<I32, I32>>& pairs);

  Vector<Contact> m_contacts;

  bool m_use_gpu = true;
  std::unique_ptr<gpu::CollisionDetector_GPU> m_gpu_detector;
};

static bool check_triangle_collision(
  const Vec3& p_local,            
  const Vec3& p0, const Vec3& p1, const Vec3& p2,
  F32& min_dist_sq,
  Vec3& out_normal,
  Vec3& out_pt
) {
  Vec3 v10 = p1 - p0;
  Vec3 v20 = p2 - p0;
  Vec3 normal = v10.cross(v20).normalized();

  F32 plane_dist = (p_local - p0).dot(normal);
  const F32 penetration_threshold = -0.5f;

  if (plane_dist < 0.0f && plane_dist > penetration_threshold) {
    F32 dist_sq = plane_dist * plane_dist;
    if (dist_sq < min_dist_sq) {
      Vec3 proj = p_local - normal * plane_dist;
      Vec3 v2 = proj - p0;
      F32 d00 = v10.dot(v10);
      F32 d01 = v10.dot(v20);
      F32 d11 = v20.dot(v20);
      F32 d20 = v2.dot(v10);
      F32 d21 = v2.dot(v20);
      F32 denom = d00 * d11 - d01 * d01;

      if (std::abs(denom) > 1e-8f) {
        F32 inv_denom = 1.0f / denom;
        F32 v = (d11 * d20 - d01 * d21) * inv_denom;
        F32 w = (d00 * d21 - d01 * d20) * inv_denom;
        F32 u = 1.0f - v - w;
        if (v >= 0.0f && w >= 0.0f && u >= 0.0f) {
          min_dist_sq = dist_sq;
          out_normal = normal;
          out_pt = proj;
          return true;
        }
      }
    }
  }
  return false;
}