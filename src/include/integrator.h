// include/integrator.h
#pragma once

#include <memory>

#include "type.h"
#include "scene.h"
#include "bvh.h"
#include "collision.h"


class ForceAssembler {
public:
  ForceAssembler();

  void setup(I32 dof_count);
  // Accumulates gravity and penalty forces into the global force vector F
  void assemble_forces(Scene& scene, const Vector<Contact>& contacts, VecX& F);
  // Accumulates stiffness matrix (dF/dx) into the global Jacobian J
  void assemble_jacobian(Scene& scene, const Vector<Contact>& contacts, SpMat& J);
  void set_gravity(const Vec3& g);
  void set_penalty_stiffness(F32 k);

private:
  void add_gravity_force(Scene& scene, VecX& F);
  void add_collision_force(Scene& scene, const Vector<Contact>& contacts, VecX& F);
  void add_collision_jacobian(const Vector<Contact>& contacts, SpMat& J);

  Vec3 m_gravity;
  F32  m_penalty_stiffness;
  F32  m_friction_stiffness;
  I32  m_dof_count;
};

class Integrator {
public:
  Integrator();

  void initialize(Scene& scene);
  void integrate(Scene& scene);
  void set_max_iterations(I32 iter);
  void set_tolerance(F32 tol);

private:
  void build_mass_matrix(Scene& scene, SpMat& M);
  void update_body_states(Scene& scene, const VecX& delta_v);
  
  bool solve(const SpMat& A, const VecX& b, VecX& x);

  CollisionSolver m_collision_solver;
  ForceAssembler  m_force_assembler;

  SpMat m_mass_matrix;   
  SpMat m_jacobian;      
  SpMat m_system_matrix; 
  VecX  m_force_vector;  
  VecX  m_delta_v;       
  VecX  m_rhs;           

  I32 m_max_iter;
  F32 m_tolerance;
  F32 m_dt;
};

inline I32 global_idx(I32 body_idx) { return body_idx * 6; }