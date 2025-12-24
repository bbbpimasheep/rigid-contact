// include/properties.h
#pragma once

#include <limits>

#include "type.h"


struct Properties {
  F32 mass                  = 1.0;
  F32 mass_inv              = 1.0;
  F32 lin_damping           = 0.4;
  F32 ang_damping           = 0.1;
  F32 restitution           = 0.0;
  F32 friction              = 0.5;

  Mat3 I_body               = Mat3::Identity();
  Mat3 I_body_inv           = Mat3::Identity();
  Vec3 com                  = Vec3::Zero();

  void set_mass(F32 value);
  void set_inertia_body(const Mat3& value);
  void set_center_of_mass(const Vec3& value);
  void finalize();
  bool is_valid() const;
};

inline F32 clamp_positive(F32 value) {
  const F32 eps = std::numeric_limits<F64>::epsilon();
  return value < eps ? eps : value;
}
inline Mat3 inverse_symmetric(const Mat3& M) {
  Eigen::LDLT<Mat3> ldlt(M);
  if (ldlt.info() == Eigen::Success)
    return ldlt.solve(Mat3::Identity());
  // fall back
  return M.completeOrthogonalDecomposition().pseudoInverse();
}