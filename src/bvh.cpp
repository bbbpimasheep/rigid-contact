#include <numeric>

#include "bvh.h"


void BVH::clear() {
  m_nodes.clear(); prim_indices.clear(); prims.clear();
  m_root_index = -1;
}
AABB BVH::accumulate_bounds(I32 first, I32 last) const {
  AABB result;
  result.reset();
  for (I32 i = first; i < last; ++i) {
    I32 prim_index = prim_indices[i];
    auto& bounds = prims[prim_index].bounds;
    result.expand(bounds.min);
    result.expand(bounds.max);
  }
  return result;
}
void BVH::centroid_bounds(I32 first, I32 last, Vec3& min_out, Vec3& max_out) const {
  min_out = vec3_pos_inf();
  max_out = vec3_neg_inf();
  for (I32 i = first; i < last; ++i) {
    I32 prim_index = prim_indices[i];
    const Vec3& centroid = prims[prim_index].centroid;
    min_out = min_out.cwiseMin(centroid);
    max_out = max_out.cwiseMax(centroid);
  }
}
I32 BVH::partition_primitives(I32 first, I32 last, I32 axis, F32 split_value) {
  auto begin_iter = prim_indices.begin() + first;
  auto end_iter   = prim_indices.begin() + last;

  auto predicate  = [&](I32 prim_index) { 
    return prims[prim_index].centroid[axis] < split_value; 
  };

  auto mid_iter   = std::partition(begin_iter, end_iter, predicate);
  return mid_iter - prim_indices.begin();
}

void BVH::build(const Vector<Vec3>& vertices,
                const Vector<Trig>& triangles) {
  clear();
  if (triangles.empty()) { return; }

  const I32 prim_count = triangles.size();
  prims.resize(prim_count);
  prim_indices.resize(prim_count);
  std::iota(prim_indices.begin(), prim_indices.end(), 0);

  for (I32 i = 0; i < prim_count; ++i) {
    const auto& tri = triangles[i];

    PrimRecord record;
    record.bounds.reset();
    record.bounds.expand(vertices[tri.x()]);
    record.bounds.expand(vertices[tri.y()]);
    record.bounds.expand(vertices[tri.z()]);
    record.centroid = (vertices[tri.x()] +
                       vertices[tri.y()] +
                       vertices[tri.z()]) * (1.0f / 3.0f);
    record.triangle_index = i;

    prims[i] = record;
  }

  m_nodes.reserve(prim_count * 2);
  m_root_index = build_node(0, prim_count, 0);
}
I32 BVH::build_node(I32 first, I32 last, I32 depth) {
  I32 node_index = m_nodes.size();
  m_nodes.emplace_back();
  BVHNode& node = m_nodes.back();

  node.bounds     = accumulate_bounds(first, last);
  node.first_prim = first;
  node.prim_count = last - first;
  node.left       = -1;
  node.right      = -1;

  const bool force_leaf = (node.prim_count <= leaf_prim_count) ||
                          (depth >= max_depth);
  if (force_leaf) { return node_index; }

  Vec3 centroid_min, centroid_max;
  centroid_bounds(first, last, centroid_min, centroid_max);
  Vec3 centroid_extent = centroid_max - centroid_min;

  I32 axis = 0;
  if (centroid_extent.y() > centroid_extent.x())   { axis = 1; }
  if (centroid_extent.z() > centroid_extent[axis]) { axis = 2; }
  if (centroid_extent[axis] <= centroid_epsilon) { return node_index; }

  F32 split_value = centroid_min[axis] + centroid_extent[axis] * 0.5f;
  I32 mid = partition_primitives(first, last, axis, split_value);

  if (mid == first || mid == last) { return node_index; }

  node.first_prim = -1;
  node.prim_count = 0;
  node.left  = build_node(first, mid, depth + 1);
  node.right = build_node(mid, last, depth + 1);

  return node_index;
}