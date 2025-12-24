// include/mesh_lib.h
#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "type.h"
#include "tri_mesh.h"

class MeshLibrary {
public:
  static MeshLibrary& instance();

  std::shared_ptr<TriMesh> acquire(const String& path,
                                   bool build_bvh = true);
  void clear();

private:
  MeshLibrary() = default;
  MeshLibrary(const MeshLibrary&)            = delete;
  MeshLibrary& operator=(const MeshLibrary&) = delete;

  using mesh_cache = std::unordered_map<String, std::weak_ptr<TriMesh>>;

  std::shared_ptr<TriMesh> load_mesh_internal(const String& path,
                                              bool build_bvh);

  std::mutex mutex;
  mesh_cache cache;
};