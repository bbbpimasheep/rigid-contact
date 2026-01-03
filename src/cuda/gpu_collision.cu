// cuda/gpu_collision.cu
#include <cstdio>

#include "gpu_collision.cuh"
#include "gpu_broadphase.cuh"

namespace gpu {

// ============== Collision Kernels ==============

__global__ void detect_vertex_plane_collision_kernel(
  const F32_3* vertices,
  I32 num_vertices,
  const BVHNode_d* nodes,
  const I32* prim_indices,
  const I32_3* triangles,
  F32_3 body_pos,
  F32_4 body_orientation,
  const Plane_d* planes,
  I32 num_planes,
  I32 body_idx,
  Contact_d* contacts,
  I32* contact_count,
  I32 max_contacts
) {
  I32 vid = blockIdx.x * blockDim.x + threadIdx.x;
  if (vid >= num_vertices) return;
  
  F32_3 v_local = vertices[vid];
  F32_3 v_world = rotate_by_quat(v_local, body_orientation) + body_pos;
    
  for (I32 p = 0; p < num_planes; ++p) {
    Plane_d plane = planes[p];
    F32 dist = dot(v_world, plane.normal) - plane.offset;
    
    if (dist < 0.0f) {
      I32 idx = atomicAdd(contact_count, 1);
      if (idx < max_contacts) {
        Contact_d contact;
        contact.body_index_a  = -1;  // Environment
        contact.body_index_b  = body_idx;
        contact.position      = v_world;
        contact.normal        = plane.normal;
        contact.depth         = -dist;
        contacts[idx]         = contact;
      }
    }
  }
}
__global__ void detect_BVH_plane_collision_kernel(
  const F32_3* vertices,
  const BVHNode_d* nodes,
  const I32* prim_indices,
  const I32_3* triangles,
  I32 num_triangles,
  F32_3 body_pos,
  F32_4 body_orientation,
  Plane_d plane_world,
  I32 body_idx,
  Contact_d* contacts,
  I32* contact_count,
  I32 max_contacts,
  bool* visited_vertices,
  I32 num_vertices
) {
  I32 leaf_id = blockIdx.x * blockDim.x + threadIdx.x;  // one leaf for one thread
  if (leaf_id >= num_triangles) return;
  
  I32 node_idx = num_triangles - 1 + leaf_id;
  BVHNode_d node = nodes[node_idx];
  
  F32_4 q_inv = make_float4(-body_orientation.x, -body_orientation.y, 
                            -body_orientation.z, body_orientation.w);
  F32_3 plane_local_normal = rotate_by_quat(plane_world.normal, q_inv);
  F32   plane_local_offset = plane_world.offset - dot(plane_world.normal, body_pos);
  
  if (!aabb_intersects_plane(node.bounds, plane_local_normal, plane_local_offset))
    return;
  
  I32 prim_idx = prim_indices[leaf_id];
  I32_3 tri = triangles[prim_idx];
  I32 vert_ids[3] = {tri.x, tri.y, tri.z};
  
  for (I32 k = 0; k < 3; ++k) {
      I32 vid = vert_ids[k];
      bool was_visited = atomicExch((I32*)&visited_vertices[vid], 1);
      if (was_visited) continue;
      
      F32_3 v_local = vertices[vid];
      F32 dist = dot(v_local, plane_local_normal) - plane_local_offset;
      
      if (dist < 0.0f) {
        F32_3 v_world = rotate_by_quat(v_local, body_orientation) + body_pos;
        I32 idx = atomicAdd(contact_count, 1);
        if (idx < max_contacts) {
          Contact_d contact;
          contact.body_index_a  = -1;
          contact.body_index_b  = body_idx;
          contact.position      = v_world;
          contact.normal        = plane_world.normal;
          contact.depth         = -dist;
          contacts[idx]         = contact;
        }
      }
  }
}

// Helper
__device__ F32_3 closest_point_on_triangle(F32_3 p, F32_3 a, F32_3 b, F32_3 c) {
  F32_3 ab = b - a;
  F32_3 ac = c - a;
  F32_3 ap = p - a;
  
  F32 d1 = dot(ab, ap);
  F32 d2 = dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) return a;
  
  F32_3 bp = p - b;
  F32 d3 = dot(ab, bp);
  F32 d4 = dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) return b;
  
  F32 vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    F32 v = d1 / (d1 - d3);
    return a + ab * v;
  }
  
  F32_3 cp = p - c;
  F32 d5 = dot(ab, cp);
  F32 d6 = dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) return c;
  
  F32 vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    F32 w = d2 / (d2 - d6);
    return a + ac * w;
  }
  
  F32 va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    F32 w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return b + (c - b) * w;
  }
  
  F32 denom = 1.0f / (va + vb + vc);
  F32 v = vb * denom;
  F32 w = vc * denom;
  return a + ab * v + ac * w;
}
// Body vs Body
__global__ void detect_body_body_collision_kernel(
  // Body A (static)
  const F32_3* vertices_a,
  const BVHNode_d* nodes_a,
  const I32* prim_indices_a,
  const I32_3* triangles_a,
  I32 num_nodes_a,
  F32_3 pos_a,
  F32_4 orientation_a,
  // Body B (dynamic)
  const F32_3* vertices_b,
  I32 num_vertices_b,
  F32_3 pos_b,
  F32_4 orientation_b,
  // Output
  I32 body_idx_a,
  I32 body_idx_b,
  Contact_d* contacts,
  I32* contact_count,
  I32 max_contacts,
  F32 collision_threshold
) {
  I32 vid = blockIdx.x * blockDim.x + threadIdx.x;
  if (vid >= num_vertices_b) return;
  
  F32_3 v_local_b = vertices_b[vid];
  F32_3 v_world = rotate_by_quat(v_local_b, orientation_b) + pos_b;
  F32_4 q_inv_a = make_float4(-orientation_a.x, -orientation_a.y, 
                              -orientation_a.z, orientation_a.w);
  F32_3 p_local_a = rotate_by_quat(v_world - pos_a, q_inv_a);
  
  I32 stack[64];
  I32 stack_ptr = 0;
  stack[stack_ptr++] = 0;  // Root
  
  F32 min_dist_sq = collision_threshold * collision_threshold;
  F32_3 closest_normal = make_float3(0, 0, 0);
  F32_3 closest_point  = make_float3(0, 0, 0);
  bool found = false;
  
  I32 num_triangles_a = (num_nodes_a + 1) / 2;
  
  while (stack_ptr > 0) {
    I32 node_idx = stack[--stack_ptr];
    BVHNode_d node = nodes_a[node_idx];
    
    F32 dist_sq = 0.0f;
    for (I32 i = 0; i < 3; ++i) {
      F32 coord = (i == 0) ? p_local_a.x : ((i == 1) ? p_local_a.y : p_local_a.z);
      F32 box_min = (i == 0) ? node.bounds.min.x : ((i == 1) ? node.bounds.min.y : node.bounds.min.z);
      F32 box_max = (i == 0) ? node.bounds.max.x : ((i == 1) ? node.bounds.max.y : node.bounds.max.z);
      
      if (coord < box_min) dist_sq += (box_min - coord) * (box_min - coord);
      else if (coord > box_max) dist_sq += (coord - box_max) * (coord - box_max);
    }
    if (dist_sq > min_dist_sq) continue;
    
    if (node.is_leaf()) {
      I32 prim_idx = prim_indices_a[node.first_prim];
      I32_3 tri = triangles_a[prim_idx];
      
      F32_3 p0 = vertices_a[tri.x];
      F32_3 p1 = vertices_a[tri.y];
      F32_3 p2 = vertices_a[tri.z];
      
      F32_3 closest = closest_point_on_triangle(p_local_a, p0, p1, p2);
      F32_3 diff = p_local_a - closest;
      F32 d_sq = dot(diff, diff);
      
      if (d_sq < min_dist_sq && d_sq > 1e-12f) {
        min_dist_sq = d_sq;
        closest_point = closest;
        // Compute normal
        F32_3 edge1 = p1 - p0;
        F32_3 edge2 = p2 - p0;
        F32_3 normal = cross(edge1, edge2);
        F32 len = sqrtf(dot(normal, normal));
        if (len > 1e-6f) {
          normal = normal * (1.0f / len);
          // make sure the normal points to the vertex
          if (dot(normal, diff) < 0) 
            normal = normal * (-1.0f);
          closest_normal = normal;
          found = true;
        }
      }
    } else {
      // internal nodes
      if (node.left >= 0 && stack_ptr < 63)  stack[stack_ptr++] = node.left;
      if (node.right >= 0 && stack_ptr < 63) stack[stack_ptr++] = node.right;
    }
  }
  
  if (found) {
      I32 idx = atomicAdd(contact_count, 1);
      if (idx < max_contacts) {
          Contact_d contact;
          contact.body_index_a  = body_idx_a;
          contact.body_index_b  = body_idx_b;
          contact.position      = v_world;
          contact.normal        = rotate_by_quat(closest_normal, orientation_a);  // to world frame
          contact.depth         = sqrtf(min_dist_sq);
          contacts[idx]         = contact;
      }
  }
}

// ============== CollisionDetector_GPU Implementation ==============

CollisionDetector_GPU::CollisionDetector_GPU() {
  cudaMalloc(&d_contacts, max_contacts * sizeof(Contact_d));
  cudaMalloc(&d_contact_count, sizeof(I32));
  m_broadphase = std::make_unique<BroadphaseGPU>();
}
CollisionDetector_GPU::~CollisionDetector_GPU() { free(); }

void CollisionDetector_GPU::free() {
  if (d_contacts)       cudaFree(d_contacts);
  if (d_contact_count)  cudaFree(d_contact_count);
  if (d_planes)         cudaFree(d_planes);
  d_contacts      = nullptr;
  d_contact_count = nullptr;
  d_planes        = nullptr;
  m_broadphase.reset();
}
void CollisionDetector_GPU::broadphase_detect(
  const Vector<Vec3>& aabb_mins,
  const Vector<Vec3>& aabb_maxs,
  Vector<CollisionPair>& out_pairs
) {
  if (m_broadphase)
    m_broadphase->detect_pairs(aabb_mins, aabb_maxs, out_pairs);
}
void CollisionDetector_GPU::set_BVH(BVHBuilder_GPU* bvh_builder) { m_bvh = bvh_builder; }
void CollisionDetector_GPU::detect_body_environment(
  I32 body_idx,
  const BodyTransform_d& transform,
  const Plane_d* planes,
  I32 num_planes,
  Vector<Contact_d>& out_contacts
) {
  if (!m_bvh || m_bvh->vertex_count() == 0) return;
  
  // Copy to GPU
  if (this->num_planes != num_planes) {
    if (d_planes) cudaFree(d_planes);
    cudaMalloc(&d_planes, num_planes * sizeof(Plane_d));
    this->num_planes = num_planes;
  }
  cudaMemcpy(d_planes, planes, num_planes * sizeof(Plane_d), cudaMemcpyHostToDevice);
  
  bool* d_visited;
  I32 num_verts = m_bvh->vertex_count();
  cudaMalloc(&d_visited, num_verts * sizeof(bool));
  cudaMemset(d_visited, 0, num_verts * sizeof(bool));
  
  I32 zero = 0;
  cudaMemcpy(d_contact_count, &zero, sizeof(I32), cudaMemcpyHostToDevice);
  
  F32_3 body_pos = make_float3(transform.position.x, transform.position.y, transform.position.z);
  F32_4 body_ori = transform.orientation;
  
  I32 blockSize = 256;
  I32 numBlocks = (m_bvh->triangle_count() + blockSize - 1) / blockSize;
  
  for (I32 p = 0; p < num_planes; ++p) {
    cudaMemset(d_visited, 0, num_verts * sizeof(bool));
    
    detect_BVH_plane_collision_kernel<<<numBlocks, blockSize>>>(
      m_bvh->device_vertices(),
      m_bvh->device_nodes(),
      m_bvh->device_prim_indices(),
      m_bvh->device_triangles(),
      m_bvh->triangle_count(),
      body_pos,
      body_ori,
      planes[p],
      body_idx,
      d_contacts,
      d_contact_count,
      max_contacts,
      d_visited,
      num_verts
    );
  }
  
  cudaFree(d_visited);
  
  // Copy back to host
  I32 contact_count;
  cudaMemcpy(&contact_count, d_contact_count, sizeof(I32), cudaMemcpyDeviceToHost);
  contact_count = min(contact_count, max_contacts);
  
  out_contacts.resize(contact_count);
  if (contact_count > 0)
    cudaMemcpy(out_contacts.data(), d_contacts, contact_count * sizeof(Contact_d), cudaMemcpyDeviceToHost);
}
void CollisionDetector_GPU::detect_body_body(
  I32 body_a_idx, const BodyTransform_d& transform_a, BVHBuilder_GPU* bvh_a,
  I32 body_b_idx, const BodyTransform_d& transform_b, BVHBuilder_GPU* bvh_b,
  Vector<Contact_d>& out_contacts
) {
  if (!bvh_a || !bvh_b) return;
  
  I32 zero = 0;
  cudaMemcpy(d_contact_count, &zero, sizeof(I32), cudaMemcpyHostToDevice);
  
  F32 collision_threshold = 0.5f;
  
  I32 blockSize = 256;
  I32 numBlocks = (bvh_b->vertex_count() + blockSize - 1) / blockSize;
  
  F32_3 pos_a = make_float3(transform_a.position.x, transform_a.position.y, transform_a.position.z);
  F32_3 pos_b = make_float3(transform_b.position.x, transform_b.position.y, transform_b.position.z);
  
  detect_body_body_collision_kernel<<<numBlocks, blockSize>>>(
    bvh_a->device_vertices(),
    bvh_a->device_nodes(),
    bvh_a->device_prim_indices(),
    bvh_a->device_triangles(),
    bvh_a->node_count(),
    pos_a,
    transform_a.orientation,
    bvh_b->device_vertices(),
    bvh_b->vertex_count(),
    pos_b,
    transform_b.orientation,
    body_a_idx,
    body_b_idx,
    d_contacts,
    d_contact_count,
    max_contacts,
    collision_threshold
  );
  
  // Opposite
  numBlocks = (bvh_a->vertex_count() + blockSize - 1) / blockSize;
  
  detect_body_body_collision_kernel<<<numBlocks, blockSize>>>(
    bvh_b->device_vertices(),
    bvh_b->device_nodes(),
    bvh_b->device_prim_indices(),
    bvh_b->device_triangles(),
    bvh_b->node_count(),
    pos_b,
    transform_b.orientation,
    bvh_a->device_vertices(),
    bvh_a->vertex_count(),
    pos_a,
    transform_a.orientation,
    body_b_idx,
    body_a_idx,
    d_contacts,
    d_contact_count,
    max_contacts,
    collision_threshold
  );
  
  // Copy back
  I32 contact_count;
  cudaMemcpy(&contact_count, d_contact_count, sizeof(I32), cudaMemcpyDeviceToHost);
  contact_count = min(contact_count, max_contacts);
  
  out_contacts.resize(contact_count);
  if (contact_count > 0)
    cudaMemcpy(out_contacts.data(), d_contacts, contact_count * sizeof(Contact_d), cudaMemcpyDeviceToHost);
}

} // namespace gpu