// include/tri_mesh.h
#pragma once

#include <memory>
#include <limits>

#include "type.h"
#include "bvh.h"


class TriMesh {
public:
  TriMesh() = default;
  explicit TriMesh(String name);

  bool load_obj(const String& path, bool buildBVH = true);
  void clear();

  const String& name() const;
  const Vector<Vec3>& vertices() const;
  const Vector<Trig>& triangles() const;
  const Vector<Vec3>& vertex_normals() const;
  const BVH& bvh() const;

  const AABB& local_bounds() const;
  void rebuild_BVH();   // TODO: narrow-phase BVH
  bool has_BVH() const;

private:
  void update_bounds();
  void mark_BVH_dirty();

  String m_name;

  Vector<Vec3> m_vertices_local;
  Vector<Trig> m_triangles;
  Vector<Vec3> m_vertex_normals;

  AABB   m_local_bounds;
  bool   m_bounds_dirty = true;

  std::unique_ptr<BVH> m_bvh;
  bool m_bvh_dirty = true;
};

inline bool start_with(const String& line, const char* prefix) {
  return line.compare(0, std::char_traits<char>::length(prefix), prefix) == 0;
}