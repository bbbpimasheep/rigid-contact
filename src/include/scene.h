#pragma once

#include <memory>

#include "type.h"
#include "rigid_body.h"


struct Plane {
  Vec3 normal = Vec3::Zero();
  F32  offset = 0.0f;
};

struct Environment {
  enum boundary_id : I32 {
    boundary_neg_x=0, boundary_pos_x,
    boundary_neg_y,   boundary_pos_y,
    boundary_neg_z,   boundary_pos_z,
    boundary_count
  };

  Environment();

  void set_aabb(const Vec3& min_corner, const Vec3& max_corner);
  Plane& plane(boundary_id id);

  Plane planes[boundary_count];
};

class Scene {
public:
  Scene() = default;

  RigidBody& create_rigid_body(const String& name);
  RigidBody* rigid_body(I32 index);
  Vector<std::unique_ptr<RigidBody>>& rigid_bodies();

  I32 rigid_body_count() const;
  void clear_bodies();
  Environment& environment();
  void set_environment(
    const Vec3& min_corner, 
    const Vec3& max_corner
  );
  bool is_empty() const;

private:
  Vector<std::unique_ptr<RigidBody>> m_rigid_bodies;
  Environment environment_desc;
};