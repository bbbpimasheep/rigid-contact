#include "scene.h"

#include <cassert>
#include <utility>


Environment::Environment() {
  Vec3 default_min(-10.0f, -10.0f, -10.0f);
  Vec3 default_max( 10.0f,  10.0f,  10.0f);
  set_aabb(default_min, default_max);
}

void Environment::set_aabb(const Vec3& min_corner, const Vec3& max_corner) {
  planes[boundary_neg_x].normal =  Vec3(1.0f, 0.0f, 0.0f);
  planes[boundary_neg_x].offset =  min_corner.x();
  planes[boundary_pos_x].normal =  Vec3(-1.0f, 0.0f, 0.0f);
  planes[boundary_pos_x].offset = -max_corner.x();
  planes[boundary_neg_y].normal =  Vec3(0.0f, 1.0f, 0.0f);
  planes[boundary_neg_y].offset =  min_corner.y();
  planes[boundary_pos_y].normal =  Vec3(0.0f, -1.0f, 0.0f);
  planes[boundary_pos_y].offset = -max_corner.y();
  planes[boundary_neg_z].normal =  Vec3(0.0f, 0.0f, 1.0f);
  planes[boundary_neg_z].offset =  min_corner.z();
  planes[boundary_pos_z].normal =  Vec3(0.0f, 0.0f, -1.0f);
  planes[boundary_pos_z].offset = -max_corner.z();
}
Plane& Environment::plane(boundary_id id) {
  assert(id >= boundary_neg_x && id < boundary_count);
  return planes[id];
}


RigidBody& Scene::create_rigid_body(const String& name) {
  m_rigid_bodies.emplace_back(std::make_unique<RigidBody>(name));
  return *m_rigid_bodies.back();
}
RigidBody* Scene::rigid_body(I32 index) {
  if (index < 0 || index >= m_rigid_bodies.size())
    return nullptr;
  return m_rigid_bodies[index].get();
}
Vector<std::unique_ptr<RigidBody>>& Scene::rigid_bodies() { return m_rigid_bodies; }
I32 Scene::rigid_body_count() const { return m_rigid_bodies.size(); }
void Scene::clear_bodies() { m_rigid_bodies.clear(); }
Environment& Scene::environment() { return environment_desc; }
void Scene::set_environment(
  const Vec3& min_corner, 
  const Vec3& max_corner) { 
  environment_desc.set_aabb(min_corner, max_corner);
}
bool Scene::is_empty() const { return m_rigid_bodies.empty(); }