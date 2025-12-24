#include "rigid_body.h"
#include "mesh_lib.h"


RigidBody::RigidBody(const String& _name) : name(_name) {}

bool RigidBody::load_mesh(const String& path, bool build_bvh) {
  auto mesh_shared = MeshLibrary::instance().acquire(path, build_bvh);
  if (!mesh_shared) return false;

  mesh_ref = std::move(mesh_shared);
  bounds_dirty  = true;
  inertia_dirty = true;
  return true;
}

TriMesh& RigidBody::mesh() { return *mesh_ref; }
void RigidBody::set_mesh(const std::shared_ptr<TriMesh>& mesh) {
  mesh_ref = mesh;
  bounds_dirty  = true;
  inertia_dirty = true;
}
bool RigidBody::has_mesh() const { return static_cast<bool>(mesh_ref); }
Properties& RigidBody::properties() { return props; }
BodyState& RigidBody::state() { return m_state; }

void RigidBody::set_properties(const Properties& new_props) {
  props = new_props;
  inertia_dirty = true;
}
void RigidBody::set_state(const BodyState& new_state) {
  m_state = new_state;
  bounds_dirty  = true;
  inertia_dirty = true;
}
AABB& RigidBody::world_bounds() {
  if (bounds_dirty) { update_world_bounds(); }
  return m_world_bounds;
}
const Mat3& RigidBody::inertia_world_inv() {
  if (inertia_dirty) { sync_inertia_world(); }
  return I_world_inv;
}
void RigidBody::apply_force(const Vec3& force, const Vec3& world_point) {
  force_accum  += force;
  Vec3 r = world_point - m_state.position;
  torque_accum += r.cross(force);
}
void RigidBody::apply_torque(const Vec3& torque) { torque_accum += torque; }
void RigidBody::clear_accumulators() {
  force_accum.setZero();
  torque_accum.setZero();
}
const Vec3& RigidBody::accumulated_force()  const { return force_accum; }
const Vec3& RigidBody::accumulated_torque() const { return torque_accum; }

bool RigidBody::is_dynamic() const { return props.mass_inv > 0.0f; }

void RigidBody::update_world_bounds() {
  m_world_bounds.reset();
  if (!mesh_ref) { bounds_dirty = false; return; }

  const auto& verts = mesh_ref->vertices();
  if (verts.empty()) { bounds_dirty = false; return; }

  Mat3 rot = m_state.orientation.toRotationMatrix();
  for (const Vec3& v_local : verts) {
    Vec3 v_center = v_local - props.com;
    Vec3 v_world  = rot * v_center + m_state.position;
    m_world_bounds.expand(v_world);
  }
  bounds_dirty = false;
}
void RigidBody::sync_inertia_world() {
  Mat3 rot = m_state.orientation.toRotationMatrix();
  I_world_inv = rot * props.I_body_inv * rot.transpose();
  inertia_dirty = false;
}