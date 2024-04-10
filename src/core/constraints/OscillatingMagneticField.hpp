
#ifndef CONSTRAINTS_OSCILLATINGMAGNETICFIELD_HPP
#define CONSTRAINTS_OSCILLATINGMAGNETICFIELD_HPP
/** \file OscillatingMagneticField.hpp
 *  Routines to calculate forses and energy for  AC Magnetic field.
 *  Field is described by vector-valued direction \vec{m}, magnitude,
 *  frequrecy and phase shift, so
 *  H = magnitude*cos(frequecy*time + phase_shift) \vec{m}
 *
 */
#include "Constraint.hpp"
#include "particle_data.hpp"


namespace Constraints {

class OscillatingMagneticField : public Constraint {
public:
  OscillatingMagneticField() : m_direction({0., 0., 1.}), m_magnitude(0.),
                      m_frequency(0.), m_phase_shift(0) {}

  void set_direction(Utils::Vector3d const &direction) { m_direction = direction; m_direction.normalize(); }

  void set_magnitude(const double &magnitude) { m_magnitude = magnitude; }

  void set_phase_shift(const double &shift) { m_phase_shift = shift; }

  void set_frequency(const double frequency) { m_frequency = frequency;}

  Utils::Vector3d  &direction()  { return m_direction; }

  double &phase_shift()  { return m_phase_shift; }

  double &frequency()  {return m_frequency; }

  double &magnitude()  {return m_magnitude; }

  void add_energy(const Particle &p, const Utils::Vector3d &,
                  double t, Observable_stat &energy) const override;

  ParticleForce force(const Particle &p, const Utils::Vector3d &, double t) override;

  bool fits_in_box(Utils::Vector3d const &box) const override { return true; }

private:
  Utils::Vector3d m_direction;
  double m_magnitude;
  double m_frequency;
  double m_phase_shift;
};

} /* namespace Constraints */

#endif
