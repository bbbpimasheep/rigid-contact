#include "properties.h"


void Properties::set_mass(F32 value) {
  mass = clamp_positive(value);
  mass_inv = 1.0 / mass;
}
void Properties::set_inertia_body(const Mat3& value) {
  I_body = value;
  I_body_inv = inverse_symmetric(I_body);
}
void Properties::set_center_of_mass(const Vec3& value) {
  com = value;
}
void Properties::finalize() {
  set_mass(mass);
  set_inertia_body(I_body);
}
bool Properties::is_valid() const {
  bool positive_mass  = mass > 0.0 && std::isfinite(mass);
  bool finite_inertia = I_body.allFinite() && I_body_inv.allFinite();
  return positive_mass && finite_inertia;
}