#include <iostream>
#include <limits>

#include "collision.h"
#include "gpu_collision.cuh"


void CollisionSolver::clear() { m_contacts.clear(); }
void CollisionSolver::detect_collisions(Scene& scene) {
  if (m_use_gpu) { 
    detect_collisions_gpu(scene); return; 
  }

  clear();
  Environment& env = scene.environment();
  I32 body_count   = scene.rigid_body_count();
  // Check Body vs Environment
  for (I32 i = 0; i < body_count; ++i) {
    RigidBody* body = scene.rigid_body(i);
    if (!body || !body->has_mesh()) continue;
    check_be(i, *body, env);
  }
  // Check Body vs Body (O(N^2) Broadphase)
  for (I32 i = 0; i < body_count; ++i) {
    RigidBody* body_a = scene.rigid_body(i);
    if (!body_a || !body_a->has_mesh()) continue;

    for (I32 j = i + 1; j < body_count; ++j) {
      RigidBody* body_b = scene.rigid_body(j);
      if (!body_b || !body_b->has_mesh()) continue;
      // Broadphase: AABB Intersection
      AABB& box_a = body_a->world_bounds();
      AABB& box_b = body_b->world_bounds();
      // Check overlap
      bool overlap = (box_a.min.array() <= box_b.max.array()).all() &&
                     (box_a.max.array() >= box_b.min.array()).all();
      if (overlap) check_bb(i, *body_a, j, *body_b);
    }
  }
}
const Vector<Contact>& CollisionSolver::contacts() const { return m_contacts; }

void CollisionSolver::check_be(I32 body_idx, RigidBody& body, Environment& env) {
  TriMesh& mesh    = body.mesh();
  BodyState& state = body.state();
  Mat3 rot = state.orientation.toRotationMatrix();
  Vec3 pos = state.position;
  
  const Vector<Vec3>& vertices = mesh.vertices();
  static Vector<bool> visited_verts; visited_verts.resize(vertices.size());

  for (I32 p_idx = 0; p_idx < Environment::boundary_count; ++p_idx) {
    Plane& plane_world = env.plane(static_cast<Environment::boundary_id>(p_idx));
    // Optimization: Check Body AABB vs Plane first
    const AABB& wb = body.world_bounds();
    // fprintf(stderr, "wb.min: %f %f %f\n", wb.min[0], wb.min[1], wb.min[2]);
    // fprintf(stderr, "wb.max: %f %f %f\n", wb.max[0], wb.max[1], wb.max[2]);
    Vec3 support_min;
    for(int k=0; k<3; ++k) support_min[k] = (plane_world.normal[k] >= 0) ? wb.min[k] : wb.max[k];
    F32 dist_min = support_min.dot(plane_world.normal) - plane_world.offset;
    
    if (dist_min > 1e-4f) continue; // Safe

    // Transform Plane to Local Space
    Plane plane_local;
    plane_local.normal = rot.transpose() * plane_world.normal;
    plane_local.offset = plane_world.offset - plane_world.normal.dot(pos);
    // fprintf(stderr, "%f %f %f\n", plane_local.normal[0], plane_local.normal[1], plane_local.normal[2]);

    if (mesh.has_BVH()) {
      std::fill(visited_verts.begin(), visited_verts.end(), false);
      // Vec3 plane_world_normal = rot * plane_local.normal;
      traverse_bvh_plane(
        mesh.bvh().node(mesh.bvh().root_index()), 
        mesh.bvh(), plane_local, vertices, mesh.triangles(), 
        rot, pos, body_idx, plane_world.normal, // plane_world_normal, 
        visited_verts
      );
    } else {
      for (const auto& v_local : vertices) {
        F32 dist_local = v_local.dot(plane_local.normal) - plane_local.offset;
        if (dist_local < 0.0f) {
          Contact contact;
          contact.body_index_a  = -1; // Environment
          contact.body_index_b  = body_idx;
          contact.depth         = -dist_local;
          contact.normal        = plane_world.normal;
          contact.position      = rot * v_local + pos;
          m_contacts.push_back(contact);
        }
      }
    }
  }
}
void CollisionSolver::check_bb(I32 idx_a, RigidBody& body_a, I32 idx_b, RigidBody& body_b) {
  // Symmetric check: Vertices of A vs Mesh B, and Vertices of B vs Mesh A
  check_vm(idx_a, body_a, idx_b, body_b);
  check_vm(idx_b, body_b, idx_a, body_a);
}
void CollisionSolver::check_vm(
  I32 idx_dynamic, RigidBody& dynamic_body,
  I32 idx_static,  RigidBody& static_body
) {
  const auto& vertices_dyn  = dynamic_body.mesh().vertices();
  const AABB& bounds_static = static_body.world_bounds();

  Mat3 rot_dyn  = dynamic_body.state().orientation.toRotationMatrix();
  Vec3 pos_dyn  = dynamic_body.state().position;
  Mat3 rot_stat = static_body.state().orientation.toRotationMatrix();
  Vec3 pos_stat = static_body.state().position;
  Mat3 rot_stat_inv = rot_stat.transpose();
  
  const TriMesh& mesh_stat = static_body.mesh();
  const auto& verts_stat = mesh_stat.vertices();
  const auto& tris_stat  = mesh_stat.triangles();

  F32 collision_threshold = 0.5f; 
  // Iterate dynamic vertices
  for (const auto& v_local : vertices_dyn) {
    Vec3 v_world = rot_dyn * v_local + pos_dyn;
    // World AABB check
    if (v_world.x() < bounds_static.min.x() || v_world.x() > bounds_static.max.x() ||
        v_world.y() < bounds_static.min.y() || v_world.y() > bounds_static.max.y() ||
        v_world.z() < bounds_static.min.z() || v_world.z() > bounds_static.max.z()) {
        continue;
    }
    // Transform point to static body's local space
    Vec3 p_local_stat = rot_stat_inv * (v_world - pos_stat);
    // Initial collision variables
    F32 min_dist_sq   = collision_threshold * collision_threshold;
    Vec3 closest_normal_local = Vec3::Zero();
    Vec3 closest_pt_local     = Vec3::Zero();
    bool found = false;
    if (mesh_stat.has_BVH()) {
      traverse_bvh_point(
        mesh_stat.bvh().node(mesh_stat.bvh().root_index()), 
        mesh_stat.bvh(), 
        p_local_stat, verts_stat, tris_stat, 
        min_dist_sq,
        closest_normal_local, 
        closest_pt_local, found
      );
    } else {
      // Brute Force Fallback (O(M))
      const AABB& lb = mesh_stat.local_bounds();
      if (p_local_stat.x() < lb.min.x() - 0.1f || p_local_stat.x() > lb.max.x() + 0.1f ||
          p_local_stat.y() < lb.min.y() - 0.1f || p_local_stat.y() > lb.max.y() + 0.1f ||
          p_local_stat.z() < lb.min.z() - 0.1f || p_local_stat.z() > lb.max.z() + 0.1f) ;
      else {
        for (auto& tri : tris_stat) {
          const Vec3& p0 = verts_stat[tri[0]];
          const Vec3& p1 = verts_stat[tri[1]];
          const Vec3& p2 = verts_stat[tri[2]];
          if (check_triangle_collision(p_local_stat, p0, p1, p2, min_dist_sq, closest_normal_local, closest_pt_local)) {
            found = true;
          }
        }
      }
    }
    if (found) {
        Contact contact;
        contact.body_index_a  = idx_static;
        contact.body_index_b  = idx_dynamic;
        contact.depth         = std::sqrt(min_dist_sq);
        contact.normal        = rot_stat * closest_normal_local;  // Local normal to World
        contact.position      = v_world;                          // The penetrating vertex
        m_contacts.push_back(contact);
    }
  }
}

void CollisionSolver::traverse_bvh_plane(
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
) {
  // Find closest point on AABB to the plane normal direction
  Vec3 center  = (node.bounds.min + node.bounds.max) * 0.5f;
  Vec3 extents = (node.bounds.max - node.bounds.min) * 0.5f;
  // Projection of extents onto plane normal
  F32 r = extents.x() * std::abs(plane_local.normal.x()) +
          extents.y() * std::abs(plane_local.normal.y()) +
          extents.z() * std::abs(plane_local.normal.z());     
  F32 s = plane_local.normal.dot(center) - plane_local.offset;
  if (s > r) return;

  if (node.is_leaf()) {
    I32 end = node.first_prim + node.prim_count;
    const auto& indices = bvh.primitive_indices();
#if defined RIGID_DEBUGGER
    // [Debug] 
    if (node.first_prim < 0 || node.first_prim >= (I32)indices.size()) {
        std::cerr << "[Error] BVH Leaf Node first_prim out of bounds! "
                  << "first_prim: " << node.first_prim 
                  << ", indices size: " << indices.size() << std::endl;
        return;
    }
    // [Debug] 
    if (end > (I32)indices.size()) {
        std::cerr << "[Warning] BVH Leaf Node prim_count overflow! "
                  << "end: " << end << ", indices size: " << indices.size() 
                  << ". Clamping to size." << std::endl;
        end = (I32)indices.size();
    }
#endif
    for (I32 i = node.first_prim; i < end; ++i) {
      I32 tri_idx = indices[i];
      const Trig& tri = triangles[tri_idx];
#if defined RIGID_DEBUGGER
      // [Debug] 
      if (tri_idx < 0 || tri_idx >= (I32)triangles.size()) {
          std::cerr << "[Error] Triangle index out of bounds! "
                    << "idx: " << tri_idx << ", triangles size: " << triangles.size() << std::endl;
          continue;
      }
#endif
      for (int k = 0; k < 3; ++k) {
        if (visited_verts[tri[k]]) continue;
        visited_verts[tri[k]] = true;
#if defined RIGID_DEBUGGER
        // [Debug] 
        if (v_idx < 0 || v_idx >= (I32)visited_verts.size()) {
             std::cerr << "[Error] Vertex index out of bounds! "
                       << "v_idx: " << v_idx << ", visited_verts size: " << visited_verts.size() << std::endl;
             continue;
        }
#endif
        const Vec3& v_local = vertices[tri[k]];
        F32 dist = v_local.dot(plane_local.normal) - plane_local.offset;
        
        if (dist < 0.0f) {
          Contact contact;
          contact.body_index_a  = -1;
          contact.body_index_b  = body_idx;
          contact.depth         = -dist;
          contact.normal        = plane_world_normal;
          contact.position      = body_rot * v_local + body_pos;
          m_contacts.push_back(contact);
        }
      }
    }
  } else {
    if (node.left != -1) 
      traverse_bvh_plane(
        bvh.node(node.left),  bvh, plane_local, vertices, triangles, body_rot, body_pos, body_idx, plane_world_normal, visited_verts
      );
    if (node.right != -1)
      traverse_bvh_plane(
        bvh.node(node.right), bvh, plane_local, vertices, triangles, body_rot, body_pos, body_idx, plane_world_normal, visited_verts
      );
  }
}
void CollisionSolver::traverse_bvh_point(
  const BVHNode& node,
  const BVH& bvh,
  const Vec3& point_local,
  const Vector<Vec3>& vertices,
  const Vector<Trig>& triangles,
  F32&  min_dist_sq,
  Vec3& closest_normal,
  Vec3& closest_pos,
  bool& found
) {
  F32 dist_sq = 0.0f;
  for(int i = 0; i < 3; ++i) {
    if(point_local[i] < node.bounds.min[i]) 
      dist_sq += (node.bounds.min[i] - point_local[i]) * 
                 (node.bounds.min[i] - point_local[i]);
    if(point_local[i] > node.bounds.max[i]) 
      dist_sq += (point_local[i] - node.bounds.max[i]) * 
                 (point_local[i] - node.bounds.max[i]);
  }
  if (dist_sq > min_dist_sq) return;
  if (node.is_leaf()) {
    I32 end = node.first_prim + node.prim_count;
    const auto& indices = bvh.primitive_indices();
    
    for (I32 i = node.first_prim; i < end; ++i) {
      const Trig& tri = triangles[indices[i]];
      const Vec3& p0 = vertices[tri[0]];
      const Vec3& p1 = vertices[tri[1]];
      const Vec3& p2 = vertices[tri[2]];

      if (check_triangle_collision(point_local, p0, p1, p2, min_dist_sq, closest_normal, closest_pos))
        found = true;
    }
  } else {
    if (node.left != -1)
      traverse_bvh_point(
        bvh.node(node.left),  bvh, point_local, vertices, triangles, min_dist_sq, closest_normal, closest_pos, found
      );
    if (node.right != -1)
      traverse_bvh_point(
        bvh.node(node.right), bvh, point_local, vertices, triangles, min_dist_sq, closest_normal, closest_pos, found
      );
  }
}

#if defined RIGID_USE_CUDA
#include "gpu_collision.cuh"
#include "gpu_broadphase.cuh"

CollisionSolver::CollisionSolver() : m_use_gpu(true) {
  m_gpu_detector = std::make_unique<gpu::CollisionDetector_GPU>();
}
CollisionSolver::~CollisionSolver() = default;

void CollisionSolver::detect_collisions_gpu(Scene& scene) {
  clear();
  Environment& env = scene.environment();
  I32 body_count   = scene.rigid_body_count();
  // Check Body vs Environment
  for (I32 i = 0; i < body_count; ++i) {
    RigidBody* body = scene.rigid_body(i);
    if (!body || !body->has_mesh()) continue;

    const BVH& bvh = body->mesh().bvh();
    if (!bvh.has_GPU_data()) continue;

    m_gpu_detector->set_BVH(bvh.gpu_builder());

    gpu::BodyTransform_d transform;
    const BodyState& state = body->state();
    transform.position = make_float3(state.position.x(), state.position.y(), state.position.z());
    transform.orientation = make_float4(
      state.orientation.x(), state.orientation.y(), 
      state.orientation.z(), state.orientation.w()
    );

    Vector<gpu::Plane_d> planes(Environment::boundary_count);
    for (int p = 0; p < Environment::boundary_count; ++p) {
      const Plane& plane = env.plane(static_cast<Environment::boundary_id>(p));
      planes[p].normal = make_float3(plane.normal.x(), plane.normal.y(), plane.normal.z());
      planes[p].offset = plane.offset;
    }

    Vector<gpu::Contact_d> gpu_contacts;
    m_gpu_detector->detect_body_environment(i, transform, planes.data(), planes.size(), gpu_contacts);
    // fprintf(stderr, "Check gpu_contacts.size():\n");
    // fprintf(stderr, "Check gpu_contacts.size(): %d\n", gpu_contacts.size());
    convert_gpu_contacts(gpu_contacts);
  }
  // Check Body vs Body
  Vector<std::pair<I32, I32>> collision_pairs;
  broadphase_gpu(scene, collision_pairs);

  for (const auto& pair : collision_pairs) {
    I32 i = pair.first;
    I32 j = pair.second;
    
    RigidBody* body_a = scene.rigid_body(i);
    RigidBody* body_b = scene.rigid_body(j);
    
    if (!body_a || !body_b) continue;
    if (!body_a->has_mesh() || !body_b->has_mesh()) continue;
    
    const BVH& bvh_a = body_a->mesh().bvh();
    const BVH& bvh_b = body_b->mesh().bvh();
    if (!bvh_a.has_GPU_data() || !bvh_b.has_GPU_data()) continue;
    
    gpu::BodyTransform_d transform_a, transform_b;
    const BodyState& state_a = body_a->state();
    const BodyState& state_b = body_b->state();
    transform_a.position    = make_float3(state_a.position.x(), state_a.position.y(), state_a.position.z());
    transform_a.orientation = make_float4(state_a.orientation.x(), state_a.orientation.y(), 
                                          state_a.orientation.z(), state_a.orientation.w());
    transform_b.position    = make_float3(state_b.position.x(), state_b.position.y(), state_b.position.z());
    transform_b.orientation = make_float4(state_b.orientation.x(), state_b.orientation.y(), 
                                          state_b.orientation.z(), state_b.orientation.w());
    Vector<gpu::Contact_d> gpu_contacts;
    m_gpu_detector->detect_body_body(
      i, transform_a, bvh_a.gpu_builder(),
      j, transform_b, bvh_b.gpu_builder(),
      gpu_contacts
    );
    convert_gpu_contacts(gpu_contacts);
  }
}
void CollisionSolver::convert_gpu_contacts(const Vector<gpu::Contact_d>& gpu_contacts) {
  for (const auto& gcontact : gpu_contacts) {
    Contact contact;
    contact.body_index_a  = gcontact.body_index_a;
    contact.body_index_b  = gcontact.body_index_b;
    contact.position      = Vec3(gcontact.position.x, gcontact.position.y, gcontact.position.z);
    contact.normal        = Vec3(gcontact.normal.x, gcontact.normal.y, gcontact.normal.z);
    contact.depth         = gcontact.depth;
    m_contacts.push_back(contact);
  }
}
void CollisionSolver::broadphase_gpu(Scene& scene, Vector<std::pair<I32, I32>>& pairs) {
  I32 body_count = scene.rigid_body_count();
  if (body_count < 2) {pairs.clear(); return; }

  Vector<Vec3> aabb_mins(body_count);
  Vector<Vec3> aabb_maxs(body_count);
  for (I32 i = 0; i < body_count; ++i) {
    RigidBody* body = scene.rigid_body(i);
    if (body && body->has_mesh()) {
      AABB& wb = body->world_bounds();
      aabb_mins[i] = wb.min;
      aabb_maxs[i] = wb.max;
    } else {
      aabb_mins[i] = Vec3( 1e30f,  1e30f,  1e30f);
      aabb_maxs[i] = Vec3(-1e30f, -1e30f, -1e30f);
    }
  }

  Vector<gpu::CollisionPair> gpu_pairs;
  m_gpu_detector->broadphase_detect(aabb_mins, aabb_maxs, gpu_pairs);

  pairs.resize(gpu_pairs.size());
  for (size_t i = 0; i < gpu_pairs.size(); ++i)
    pairs[i] = std::make_pair(gpu_pairs[i].body_a, gpu_pairs[i].body_b);
}

#else

CollisionSolver::CollisionSolver() : m_use_gpu(false) {}
CollisionSolver::~CollisionSolver() = default;
void CollisionSolver::detect_collisions_gpu(Scene& scene) { detect_collisions(scene); }
void CollisionSolver::convert_gpu_contacts(const Vector<gpu::Contact_d>&) {}

#endif