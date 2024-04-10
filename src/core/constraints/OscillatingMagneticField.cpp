#include <cmath>
#include "OscillatingMagneticField.hpp"
#include "energy.hpp"

namespace Constraints {

ParticleForce OscillatingMagneticField::force(const Particle &p,
                                              const Utils::Vector3d &,
                                              double t) {
#if defined(ROTATION) && defined(DIPOLES)
  Utils::Vector3d field = std::cos(m_frequency*t + m_phase_shift)*m_magnitude*m_direction;
  return {Utils::Vector3d{}, vector_product(p.calc_dip(), field)};
#else
  return {Utils::Vector3d{}};
#endif
}

void OscillatingMagneticField::add_energy(const Particle &p,
                                          const Utils::Vector3d &,
                                          double t,
                                          Observable_stat &energy) const {
#ifdef DIPOLES
  Utils::Vector3d field = std::cos(m_frequency*t + m_phase_shift)*m_magnitude*m_direction;
  energy.dipolar[0] += -1.0 * field * p.calc_dip();
#endif
}

} // namespace Constraints
