/*
 * Copyright (C) 2010-2022 The ESPResSo project
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010
 *   Max-Planck-Institute for Polymer Research, Theory Group
 *
 * This file is part of ESPResSo.
 *
 * ESPResSo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESPResSo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef ESPRESSO_SRC_CORE_MAGNETOSTATICS_DIPOLAR_DIRECT_SUM_HPP
#define ESPRESSO_SRC_CORE_MAGNETOSTATICS_DIPOLAR_DIRECT_SUM_HPP

#include "config/config.hpp"

#ifdef DIPOLES

#include "ParticleRange.hpp"

/**
 * @brief Dipolar all with all and no replica.
 * Handling of a system of dipoles where no replicas exist.
 * Assumes minimum image convention for those axis in which the
 * system is periodic.
 */
struct DipolarDirectSum {
  double prefactor;
  int n_replicas;
  DipolarDirectSum(double prefactor, int n_replicas);

  void on_activation() const {}
  void on_boxl_change() const {}
  void on_node_grid_change() const {}
  void on_periodicity_change() const {}
  void on_cell_structure_change() const {}
  void init() const {}
  void sanity_checks() const {}

  double long_range_energy(ParticleRange const &particles) const;
  void add_long_range_forces(ParticleRange const &particles) const;
  void dipole_field_at_part(ParticleRange const &particles) const;
  double funct(double theta, double h, double phi0, double kT_KVm_inv,
               double tau0_inv, double dt) const;
  /* Stoner-Wolfarth with kinetic MC step approach. For every integration
   * step we reset the dipole moment of all magnetic paricles in the simulation
   * box. Requires ESPRESSO_BUILD_WITH_NLOPT=ON and compiles in with #DIPSUS
   * feature enabled In order for the procedure to work, various properties are
   * required to be set on a per particle:
   * &Prop::sw_real marks part carrying the director
   * &Prop::sw_virt marks the particle carrying the magentic properties
   * &Prop::phi0 stores last known phi value (imposes locality)
   * &Prop::sat_mag store the saturation magnetisation value in REDUCED UNITS!
   * &Prop::Hkinv B field scale [kg/ A s^2 (T)] divided by the anisotropy field
   * &Prop::kT_KVm_inv in SI UNTIS!
   * &Prop::tau0_inv the attempt frequency, taken as the temperature in SI units
   * dependant Brown factor inverse
   * &Prop::dt_incr real time passed in a timestep given in SI untis
   */
  void stoner_wolfarth_main(ParticleRange const &particles) const;
};

#endif // DIPOLES
#endif
