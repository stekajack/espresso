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

#include "config/config.hpp"

#ifdef DIPOLES
#define TWO_M_PI 2 * M_PI

#include "magnetostatics/dipolar_direct_sum.hpp"

#include "cells.hpp"
#include "communication.hpp"
#include "constraints.hpp"
#include "constraints/HomogeneousMagneticField.hpp"
#include "errorhandling.hpp"
#include "grid.hpp"

#include <utils/cartesian_product.hpp>
#include <utils/constants.hpp>
#include <utils/math/sqr.hpp>
#include <utils/mpi/iall_gatherv.hpp>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/range/counting_range.hpp>

#include <nlopt.hpp>

#include "event.hpp"
#include "magnetostatics/stoner_wolfarth_thermal.hpp"
#include "rotation.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <mpi.h>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {
double phi_objective(unsigned n, const double *x, double *grad,
                     void *my_func_data) {
  double phi = x[0];
  double *params = (double *)my_func_data;
  double theta = params[0];
  double h = params[1];
  if (grad) {
    grad[0] = 0.5 * std::sin(2 * (phi - theta)) + h * std::sin(phi);
  }
  return -0.25 - 0.25 * std::cos(2 * (phi - theta)) - h * std::cos(phi);
}

/*
SW energy minimisation step with kinetic MC step. SW energy normalised by the
anisotropy field H_k (Hkinv stored on part to avoid division, h is the reduced
field  due to the normalisation)
*/
double funct(double theta, double h, double phi0, double kT_KVm_inv,
             double tau0_inv, double dt) {
  std::random_device rd;
  std::default_random_engine generator(rd());
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  double eps_phi = 1e-3;

  nlopt::opt opt(nlopt::LD_MMA, 1);
  double params[] = {theta, h};

  opt.set_min_objective(phi_objective, &params);
  opt.set_ftol_rel(
      1e-15); // Set the relative tolerance for the objective function value
  opt.set_ftol_abs(
      1e-15); // Set the relative tolerance for the objective function value
  std::vector<double> x(1);

  x[0] = phi0 + eps_phi; /* make initial guess from previos position plus
                                   an arbitrary perturbation*/
  double min1; /* this is the actuall value of the energy from minimiser */
  opt.optimize(x, min1);
  double phi_min1 = fmod(x[0], TWO_M_PI);

  x[0] = fmod(phi_min1 + M_PI + eps_phi, 2 * M_PI);
  /*try to find another minimimum from the other side*/
  double min2;
  opt.optimize(x, min2);

  double phi_min2 = fmod(x[0], TWO_M_PI);
  double sol;

  /* If there more that one minimum in the U_{SW} run kinetic MC step. Find
   * U_{SW} maxima to calculate the barried height for the jump probabilities.
   * Same logic as before, same issues */

  if (fabs(phi_min1 - phi_min2) > 1.e-3) {
    opt.set_max_objective(phi_objective, &params);
    x[0] = phi0 + eps_phi;

    double max1;
    opt.optimize(x, max1);
    double phi_max1 = fmod(x[0], TWO_M_PI);
    x[0] = fmod(phi_max1 + M_PI, 2 * M_PI);
    double max2;
    opt.optimize(x, max2);

    double b1 = abs(max1 - min1) * kT_KVm_inv;
    double b2 = abs(max2 - min1) * kT_KVm_inv;

    double tau1_inv = tau0_inv * exp(-b1);
    double tau2_inv = tau0_inv * exp(-b2);

    //  a multiplicative factor p0 asumed to be 1!!!
    double p12 = 0.5 * (2. - exp(-dt * tau1_inv) - exp(-dt * tau2_inv));

    if (distribution(generator) < p12) {
      sol = phi_min2;
    } else {
      sol = phi_min1;
    }
  } else {
    sol = phi_min1;
  }
  return fmod(sol + TWO_M_PI, TWO_M_PI);
}

} // namespace

void stoner_wolfarth_main(ParticleRange const &particles) {
  /* collect particle data */
  std::vector<Particle *> local_real_particles;
  std::vector<Particle *> local_virt_particles;
  local_real_particles.reserve(particles.size());
  local_virt_particles.reserve(particles.size());
  for (auto &p : particles) {
    if (p.sw_real() == 1) {
      local_real_particles.emplace_back(&p);
    } else if (p.sw_virt() == 1) {
      local_virt_particles.emplace_back(&p);
    }
  }
  // must assert that there is an equal number of sw_reals and sw_virts
  Utils::Vector3d cntrl = {0., 0., 0.};
  Utils::Vector3d ext_fld = {0., 0., 0.};
  /* collect HomogeneousMagneticFields if active */
  for (auto const &constraint : ::Constraints::constraints) {
    auto ptr = dynamic_cast<::Constraints::HomogeneousMagneticField *const>(
        &*constraint);
    if (ptr != nullptr) {
      ext_fld += ptr->H();
    }
  }
  if (ext_fld != cntrl) {
    auto p = local_virt_particles.begin();
    for (auto pi = local_real_particles.begin();
         pi != local_real_particles.end(); ++pi, ++p) {

      ext_fld += (*p)->dip_fld();

      double h = ext_fld.norm() * (*p)->Hkinv();
      auto e_h = ext_fld.normalized();
      // calc_director() result already normalised
      Utils::Vector3d e_k = (*pi)->calc_director();
      double theta = std::acos(e_h * e_k);
      if (theta > M_PI_2) {
        theta = M_PI - theta;
        e_h = -e_h;
      }
      auto rot_axis =
          vector_product(vector_product(e_h, e_k), e_h).normalized();
      auto phi = funct(theta, h, (*pi)->phi0(), (*pi)->kT_KVm_inv(),
                       (*pi)->tau0_inv(), (*pi)->dt_incr());
      (*pi)->phi0() = phi;
      auto mom = e_h * std::cos(phi) + rot_axis * std::sin(phi);
      auto const [quat, dipm] = convert_dip_to_quat(mom * (*p)->sat_mag());
      (*p)->dipm() = dipm;
      (*p)->quat() = quat;
    }
    on_dipoles_change();
  } else {

    auto p = local_virt_particles.begin();
    for (auto pi = local_real_particles.begin();
         pi != local_real_particles.end(); ++pi, ++p) {
      auto phi = fmod(funct(0., 0., (*pi)->phi0(), (*pi)->kT_KVm_inv(),
                            (*pi)->tau0_inv(), (*pi)->dt_incr()),
                      TWO_M_PI);
      Utils::Vector3d e_k = (*pi)->calc_director();
      if (phi < M_PI_2) {
        auto const [quat, dipm] = convert_dip_to_quat((*p)->sat_mag() * e_k);
        (*p)->dipm() = dipm;
        (*p)->quat() = quat;
        (*pi)->phi0() = 0.;
      } else {
        auto const [quat, dipm] = convert_dip_to_quat((*p)->sat_mag() * -e_k);
        (*p)->dipm() = dipm;
        (*p)->quat() = quat;
        (*pi)->phi0() = M_PI;
      }
    }
    on_dipoles_change();
  }
}
#endif