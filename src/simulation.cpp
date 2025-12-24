#include "simulation.h"


void Simulation::initialize() {
  m_integrator.initialize(m_scene);
  m_frame_count = 0;
}
void Simulation::reset() {
  m_scene.clear_bodies();
  m_frame_count = 0;
}
void Simulation::step() {
  m_integrator.integrate(m_scene);
  m_frame_count++;
}
void Simulation::set_environment(
  const Vec3& min_corner, 
  const Vec3& max_corner) {
  m_scene.set_environment(min_corner, max_corner);
}
void Simulation::set_iterations(I32 iter) { m_integrator.set_max_iterations(iter); }
void Simulation::set_tolerance(F32 tol) { m_integrator.set_tolerance(tol); }
void Simulation::export_frame(const String& filename) { write_scene(m_scene, filename); }
Scene& Simulation::scene() { return m_scene; }
I32 Simulation::current_frame() const { return m_frame_count; }

RigidBody& Simulation::add_body(const String& name, const String& mesh_path, F32 mass) {
  RigidBody& body = m_scene.create_rigid_body(name);

  auto mesh_ptr = MeshLibrary::instance().acquire(mesh_path, true);
  body.set_mesh(mesh_ptr);

  Properties props;
  props.set_mass(mass);
  props.finalize(); 
  body.set_properties(props);

  return body;
}