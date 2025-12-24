#include "mesh_lib.h"


MeshLibrary& MeshLibrary::instance() {
  static MeshLibrary inst;
  return inst;
}

std::shared_ptr<TriMesh> MeshLibrary::acquire(const String& path,
                                              bool build_bvh) {
  std::lock_guard<std::mutex> lock(mutex);

  auto iter = cache.find(path);
  if (iter != cache.end()) {
    if (auto shared = iter->second.lock())
      return shared;
  }

  auto mesh = load_mesh_internal(path, build_bvh);
  if (mesh) cache[path] = mesh;
  return mesh;
}

void MeshLibrary::clear() {
  std::lock_guard<std::mutex> lock(mutex);
  cache.clear();
}

std::shared_ptr<TriMesh> MeshLibrary::load_mesh_internal(
  const String& path, bool build_bvh) {

  auto mesh = std::make_shared<TriMesh>();
  if (!mesh->load_obj(path, build_bvh))
    return nullptr;
  
  return mesh;
}