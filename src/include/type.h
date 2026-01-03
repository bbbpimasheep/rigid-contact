// include/type.h
#pragma once
#include <vector>
#include <string>

// tbb
#include "oneapi/tbb.h"
#include "oneapi/tbb/parallel_reduce.h"
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/concurrent_vector.h"
#include "oneapi/tbb/concurrent_unordered_map.h"
// Eigen
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "Eigen/Sparse"

#define RIGID_DEBUGGER
#undef  RIGID_DEBUGGER

#define RIGID_USE_CUDA
// #undef  RIGID_USE_CUDA

using I32 = int;
using I64 = long int;
using F32 = float;
using F64 = double;

#if defined RIGID_USE_CUDA
#include <cuda_runtime.h>
using I32_3 = int3;
using F32_3 = float3;
using F32_4 = float4;
#endif
using U_I32 = unsigned int;

template <typename T>
using Vector = std::vector<T>;
using String = std::string;

using Vec3  = Eigen::Vector3f;
using Mat3  = Eigen::Matrix3f;
using VecX  = Eigen::VectorXf;
using Trig  = Eigen::Vector3i;
using Quat  = Eigen::Quaternionf;
using SpMat = Eigen::SparseMatrix<float>;
using Trips = std::vector<Eigen::Triplet<float>>;

struct AABB {
  Vec3 min;
  Vec3 max;

  AABB() { reset(); };
  void reset() {
    const F32 pos_inf = std::numeric_limits<F32>::infinity();
    const F32 neg_inf = -pos_inf;
    min = Vec3(pos_inf, pos_inf, pos_inf);
    max = Vec3(neg_inf, neg_inf, neg_inf);
  }
  void expand(const Vec3& p) {
    min = min.cwiseMin(p);
    max = max.cwiseMax(p);
  }
  bool is_valid() const { return (min.array() <= max.array()).all(); }
};