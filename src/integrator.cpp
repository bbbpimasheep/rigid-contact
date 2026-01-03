#include <iostream>
#include <limits>

#include "integrator.h"


ForceAssembler::ForceAssembler() 
  : m_gravity(0.0f, 0.0f, -9.81f)
  , m_penalty_stiffness(100.f)  // Increased stiffness for body-body
  , m_friction_stiffness(0.1f)
  , m_dof_count(0)
{}

void ForceAssembler::setup(I32 dof_count) { m_dof_count = dof_count; }
void ForceAssembler::set_gravity(const Vec3& g) { m_gravity = g; }
void ForceAssembler::set_penalty_stiffness(F32 k) { m_penalty_stiffness = k; }
void ForceAssembler::assemble_forces(Scene& scene, const Vector<Contact>& contacts, VecX& force) {
  force.setZero();
  add_gravity_force(scene, force);
  add_collision_force(scene, contacts, force);
}
void ForceAssembler::add_gravity_force(Scene& scene, VecX& force) {
  I32 n_bodies = scene.rigid_body_count();
  for (I32 i = 0; i < n_bodies; ++i) {
    RigidBody* body = scene.rigid_body(i);
    if (!body || !body->is_dynamic()) continue;

    I32 idx = global_idx(i);
    F32 mass = body->properties().mass;
    force.segment<3>(idx) += mass * m_gravity;
  }
}
void ForceAssembler::add_collision_force(Scene& scene, const Vector<Contact>& contacts, VecX& force) {
  for (const auto& contact : contacts) {
    Vec3 penalty = m_penalty_stiffness * contact.depth * contact.depth * contact.normal;
    // Apply to Body B (The penetrator) -> Pushed OUT (direction of normal)
    if (contact.body_index_b >= 0) {
      RigidBody* body_b = scene.rigid_body(contact.body_index_b);
      if (body_b && body_b->is_dynamic()) {
        I32 idx_b = global_idx(contact.body_index_b);
        force.segment<3>(idx_b) += penalty;
        // torque
        Vec3 r_b = contact.position - body_b->state().position;
        force.segment<3>(idx_b + 3) += r_b.cross(penalty);
      }
    }
    // Apply to Body A (The obstacle) -> Pushed IN (opposite to normal)
    if (contact.body_index_a >= 0) {
      RigidBody* body_a = scene.rigid_body(contact.body_index_a);
      if (body_a && body_a->is_dynamic()) {
        I32 idx_a = global_idx(contact.body_index_a);
        force.segment<3>(idx_a) -= penalty;
        // torque
        Vec3 r_a = contact.position - body_a->state().position;
        force.segment<3>(idx_a + 3) += r_a.cross(-penalty);
      }
    }
  }
}
void ForceAssembler::assemble_jacobian(Scene& scene, const Vector<Contact>& contacts, SpMat& Jacobian) {
  add_collision_jacobian(contacts, Jacobian);
}
void ForceAssembler::add_collision_jacobian(const Vector<Contact>& contacts, SpMat& Jacobian) {
  Trips triplets;
  
  for (const auto& contact : contacts) {
    Mat3 P_n = contact.normal * contact.normal.transpose();
    Mat3 P_t = Mat3::Identity() - P_n;
    Mat3 K_linear = -m_penalty_stiffness * P_n - m_friction_stiffness * P_t;
    // Jacobian for Body B (Penetrator)
    if (contact.body_index_b >= 0) {
      I32 idx_b = global_idx(contact.body_index_b);
      for (int r = 0; r < 3; ++r) 
        for (int k = 0; k < 3; ++k) 
          triplets.push_back(Eigen::Triplet<float>(idx_b + r, idx_b + k, K_linear(r, k)));
    }
    // Jacobian for Body A (Obstacle)
    if (contact.body_index_a >= 0) {
      I32 idx_a = global_idx(contact.body_index_a);
      for (int r = 0; r < 3; ++r)
        for (int k = 0; k < 3; ++k)
          triplets.push_back(Eigen::Triplet<float>(idx_a + r, idx_a + k, K_linear(r, k)));
    }
  }
  Jacobian.setFromTriplets(triplets.begin(), triplets.end());
}


Integrator::Integrator() 
  : m_max_iter(20) , m_tolerance(1e-2f), m_dt(1e-2f) {}

void Integrator::initialize(Scene& scene) {
  I32 n_bodies = scene.rigid_body_count();
  I32 dof = n_bodies * 6;
  
  m_force_assembler.setup(dof);

  m_mass_matrix.resize(dof, dof);
  m_jacobian.resize(dof, dof);
  m_system_matrix.resize(dof, dof);
  m_force_vector.resize(dof);
  m_delta_v.resize(dof);
  m_rhs.resize(dof);
}
void Integrator::set_max_iterations(I32 iter) { m_max_iter = iter; }
void Integrator::set_tolerance(F32 tol) { m_tolerance = tol; }
void Integrator::build_mass_matrix(Scene& scene, SpMat& M) {
  Trips triplets;
  I32 n_bodies = scene.rigid_body_count();
  
  for (I32 i = 0; i < n_bodies; ++i) {
    RigidBody* body = scene.rigid_body(i);
    if (!body) continue;
    
    I32 idx = global_idx(i);
    if (!body->is_dynamic()) {
      for (int k = 0; k < 6; ++k) {
        triplets.push_back(Eigen::Triplet<float>(idx + k, idx + k, 1e10f));
      }
    } else {
      F32 m = body->properties().mass;
      for (int k = 0; k < 3; ++k) {
        triplets.push_back(Eigen::Triplet<float>(idx + k, idx + k, m));
      }
      Mat3 I_inv = body->inertia_world_inv();
      Mat3 I = I_inv.inverse(); 
      
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          triplets.push_back(Eigen::Triplet<float>(idx + 3 + r, idx + 3 + c, I(r, c)));
        }
      }
    }
  }
  M.setFromTriplets(triplets.begin(), triplets.end());
}
void Integrator::integrate(Scene& scene) {
  I32 n_bodies = scene.rigid_body_count();
  I32 dof = n_bodies * 6;
  if (m_mass_matrix.rows() != dof) { initialize(scene); }

  for(I32 i=0; i<n_bodies; ++i)
    if(auto body = scene.rigid_body(i)) 
      body->world_bounds();
  m_collision_solver.set_use_GPU(true);
  m_collision_solver.detect_collisions(scene);

  m_mass_matrix.setZero();
  build_mass_matrix(scene, m_mass_matrix);
  
  std::cout << "[SIM_INFO] Contact number " << m_collision_solver.contacts().size() << std::endl;

  m_force_assembler.assemble_forces(scene, m_collision_solver.contacts(), m_force_vector);
  m_jacobian.setZero();
  m_force_assembler.assemble_jacobian(scene, m_collision_solver.contacts(), m_jacobian);
  
  // (M - h^2 * K) * delta_v = h * f
  SpMat stiffness_term = m_jacobian * (m_dt * m_dt);
  m_system_matrix = m_mass_matrix - stiffness_term;
  
  m_rhs = m_force_vector * m_dt;
  
  m_delta_v.setZero();
  if (solve(m_system_matrix, m_rhs, m_delta_v))
    update_body_states(scene, m_delta_v);
}
bool Integrator::solve(const SpMat& A, const VecX& b, VecX& x) {
  Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper> solver;
  solver.setMaxIterations(m_max_iter);
  solver.setTolerance(m_tolerance);
  solver.compute(A);
  if (solver.info() != Eigen::Success) return false;
  x = solver.solve(b);
  return (solver.info() == Eigen::Success);
}
void Integrator::update_body_states(Scene& scene, const VecX& delta_v) {
  I32 n_bodies = scene.rigid_body_count();

  for (I32 i = 0; i < n_bodies; ++i) {
  // tbb::parallel_for(
  //   0, n_bodies, 1, [&](I32 i) {
      RigidBody* body = scene.rigid_body(i);
      if (body && body->is_dynamic()) {
        I32 idx = global_idx(i);

        Vec3 dv_lin = delta_v.segment<3>(idx);
        Vec3 dv_ang = delta_v.segment<3>(idx + 3);
        
        BodyState& state = body->state();
        
        state.lin_velocity += dv_lin;
        state.ang_velocity += dv_ang;

        auto& props = body->properties();
        state.lin_velocity *= (1.0f - props.lin_damping * m_dt);
        state.ang_velocity *= (1.0f - props.ang_damping * m_dt);

        state.position += state.lin_velocity * m_dt;
        // fprintf(stderr, "%f %f %f\n", state.position[0], state.position[1], state.position[2]);
        
        Vec3 w = state.ang_velocity;
        Quat q_w(0, w.x(), w.y(), w.z());
        Quat dq = q_w * state.orientation;
        
        state.orientation.coeffs() += dq.coeffs() * (0.5f * m_dt);
        state.orientation.normalize();
        
        body->set_state(state);
        body->clear_accumulators();
      }
    }
  // );
}