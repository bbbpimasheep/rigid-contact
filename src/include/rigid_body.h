// include/rigid_body.h
#pragma once

#include <memory>
#include <utility>
#include <cmath>

#include "type.h"
#include "tri_mesh.h"
#include "properties.h"


struct BodyState {
  Vec3 position         = Vec3::Zero();
  Quat orientation      = Quat::Identity();
  Vec3 lin_velocity     = Vec3::Zero();
  Vec3 ang_velocity     = Vec3::Zero();
};

class RigidBody {
public:
  RigidBody() = default;
  explicit RigidBody(const String& _name);

  bool load_mesh(const String& path, bool build_bvh = true);

  TriMesh& mesh();
  void set_mesh(const std::shared_ptr<TriMesh>& mesh);
  bool has_mesh() const;
  Properties& properties();
  BodyState& state();
  AABB& world_bounds();

  void set_properties(const Properties& new_props);
  void set_state(const BodyState& new_state);
  void apply_force(const Vec3& force, const Vec3& world_point);
  void apply_torque(const Vec3& torque);
  void clear_accumulators();

  const Mat3& inertia_world_inv();
  const Vec3& accumulated_force() const;
  const Vec3& accumulated_torque() const;

  bool is_dynamic() const;

private:
  void update_world_bounds();
  void sync_inertia_world();

  std::shared_ptr<TriMesh> mesh_ref;
  Properties props;
  BodyState m_state;

  Vec3 force_accum  = Vec3::Zero();
  Vec3 torque_accum = Vec3::Zero();

  AABB m_world_bounds;
  Mat3 I_world_inv = Mat3::Identity();

  bool bounds_dirty = true;
  bool inertia_dirty = true;

  String name;
};