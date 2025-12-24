// include/simulation.h
#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "type.h"
#include "scene.h"
#include "integrator.h"
#include "mesh_lib.h"


class Simulation {
public:
  Simulation() = default;

  void initialize();
  void reset();
  void step();

  RigidBody& add_body(const String& name, const String& mesh_path, F32 mass = 1.0f);
  void set_environment(const Vec3& min_corner, const Vec3& max_corner);
  void set_iterations(I32 iter);
  void set_tolerance(F32 tol);
  void export_frame(const String& filename);

  Scene& scene();
  I32 current_frame() const;

private:
  Scene      m_scene;
  Integrator m_integrator;
  I32        m_frame_count = 0;
};

inline bool write_scene(Scene& scene, const String& filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return false;
  }

  const I32 n = scene.rigid_body_count();

  std::vector<size_t> vcount(n, 0);
  std::vector<size_t> voffset(n, 0);
  std::vector<bool>   valid(n, false);

  size_t running = 1; // OBJ index starts at 1
  for (I32 i = 0; i < n; ++i) {
    RigidBody* body = scene.rigid_body(i);
    if (!body || !body->has_mesh()) {
      valid[i] = false;
      voffset[i] = running;
      continue;
    }
    const TriMesh& mesh = body->mesh();
    valid[i] = true;
    vcount[i] = mesh.vertices().size();
    voffset[i] = running;
    running += vcount[i];
  }

  std::vector<std::string> blocks(n);

  tbb::parallel_for(tbb::blocked_range<I32>(0, n), [&](const tbb::blocked_range<I32>& r) {
    for (I32 i = r.begin(); i != r.end(); ++i) {
      if (!valid[i]) continue;

      RigidBody* body = scene.rigid_body(i);
      if (!body || !body->has_mesh()) continue;

      const TriMesh&   mesh  = body->mesh();
      const BodyState  state = body->state();

      const auto& verts = mesh.vertices();
      const auto& tris  = mesh.triangles();
      const size_t offset = voffset[i];

      std::ostringstream ss;
      ss.setf(std::ios::fmtflags(0), std::ios::floatfield);
      ss.precision(9);

      // name
      ss << "o " << mesh.name() << "_" << i << "\n";
      // v_world = R * v_local + p
      for (const auto& v_local : verts) {
        Vec3 v_world = state.orientation * v_local + state.position;
        ss << "v " << v_world.x() << " " << v_world.y() << " " << v_world.z() << "\n";
      }
      // faces (OBJ is 1-based)
      for (const auto& t : tris) 
        ss << "f " << (t[0] + offset) << " "
                   << (t[1] + offset) << " "
                   << (t[2] + offset) << "\n";

      blocks[i] = ss.str();
    }
  });

  for (I32 i = 0; i < n; ++i) 
    if (valid[i]) file << blocks[i];

  return true;
}