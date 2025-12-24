#include <fstream>
#include <sstream>
#include <iterator>

#include "tri_mesh.h"


TriMesh::TriMesh(String name) : m_name(std::move(name)) {}

bool TriMesh::load_obj(const String& path, bool buildBVH) {
  std::ifstream in(path);
  if (!in) { return false; }

  clear();

  String line;
  while (std::getline(in, line)) {
    std::istringstream iss(line);
    String tag;
    if (!(iss >> tag)) { continue; }

    if (tag == "v") {
      F32 x, y, z;
      iss >> x >> y >> z;
      m_vertices_local.emplace_back(x, y, z);
    } else if (tag == "vn") {
      F32 x, y, z;
      iss >> x >> y >> z;
      m_vertex_normals.emplace_back(x, y, z);
    } else if (tag == "f") {
      Vector<I32> indices;
      String vert_token;
      while (iss >> vert_token) {
        std::istringstream vs(vert_token);
        String item;
        std::getline(vs, item, '/');
        if (item.empty()) {  continue; }
        indices.push_back(static_cast<I32>(std::stoi(item)) - 1);
      }

      if (indices.size() < 3) { continue; }

      for (size_t i = 1; i + 1 < indices.size(); ++i) {
        m_triangles.emplace_back(
          indices[0], indices[i], indices[i + 1]
        );
      }
    }
  }

  update_bounds();
  mark_BVH_dirty();

  if (buildBVH) { rebuild_BVH(); }
  return true;
}
void TriMesh::clear() {
  m_vertices_local.clear();
  m_triangles.clear();
  m_vertex_normals.clear();
  m_local_bounds.reset();
  m_bounds_dirty = true;
  m_bvh.reset();
  m_bvh_dirty = true;
}
const String& TriMesh::name()                 const { return m_name; }
const Vector<Vec3>& TriMesh::vertices()       const { return m_vertices_local; }
const Vector<Trig>& TriMesh::triangles()      const { return m_triangles; }
const Vector<Vec3>& TriMesh::vertex_normals() const { return m_vertex_normals; }
const AABB& TriMesh::local_bounds()           const { return m_local_bounds; }
bool TriMesh::has_BVH() const { return m_bvh && m_bvh->root_index() >= 0 && !m_bvh->nodes().empty(); }
void TriMesh::mark_BVH_dirty() { m_bvh_dirty = true; }
const BVH&  TriMesh::bvh() const { 
  static const BVH empty_bvh; 
  if (m_bvh) return *m_bvh;
  return empty_bvh;
}
void TriMesh::update_bounds() {
  m_local_bounds.reset();
  for (auto& v : m_vertices_local) {
    m_local_bounds.expand(v);
  }
  m_bounds_dirty = false;
}
void TriMesh::rebuild_BVH() {
  // TODO: implement BVH construction
  if (m_bounds_dirty) { update_bounds(); }
  if (!m_bvh) { m_bvh = std::make_unique<BVH>(); }

  m_bvh->build(m_vertices_local, m_triangles);
  m_bvh_dirty = false;
}