// include/bvh.h
#pragma once

#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>

#include "type.h"

namespace gpu { class BVHBuilder_GPU; }


struct BVHNode {
  AABB bounds;
  I32 left  = -1;
  I32 right = -1;
  I32 first_prim = -1;
  I32 prim_count = 0;

  bool is_leaf() const { return prim_count > 0; }
};

class BVH {
public:
  BVH();
  ~BVH();

  void build(const Vector<Vec3>& vertices, const Vector<Trig>& triangles);
  void buildGPU(const Vector<Vec3>& vertices, const Vector<Trig>& triangles);
  void clear();
  gpu::BVHBuilder_GPU* gpu_builder() const { return m_gpu_builder.get(); }
  bool has_GPU_data() const { return m_gpu_builder != nullptr; }
  bool is_built()  const { return m_root_index >= 0; }
  I32 root_index() const { return m_root_index; }
  I32 node_count() const { return m_nodes.size(); }

  const BVHNode& node(I32 index) const { return m_nodes[index]; }
  const Vector<BVHNode>& nodes() const { return m_nodes; }
  const Vector<I32>& primitive_indices() const { return prim_indices; }

private:
  struct PrimRecord {
    AABB bounds;
    Vec3 centroid;
    I32 triangle_index = -1;
  };

  I32 build_node(I32 first, I32 last, I32 depth);
  AABB accumulate_bounds(I32 first, I32 last) const;
  void centroid_bounds(I32 first, I32 last, Vec3& min_out, Vec3& max_out) const;
  I32 partition_primitives(I32 first, I32 last, I32 axis, F32 split_value);

  Vector<BVHNode> m_nodes;
  Vector<I32> prim_indices;
  Vector<PrimRecord> prims;
  I32 m_root_index        = -1;
  I32 leaf_prim_count     = 4;
  I32 max_depth           = 64;
  F32 centroid_epsilon    = 1e-4;

  std::unique_ptr<gpu::BVHBuilder_GPU> m_gpu_builder;
};

inline Vec3 vec3_pos_inf() {
    F32 inf = std::numeric_limits<F32>::infinity();
    return Vec3(inf, inf, inf);
}
inline Vec3 vec3_neg_inf() {
    F32 inf = std::numeric_limits<F32>::infinity();
    return Vec3(-inf, -inf, -inf);
}