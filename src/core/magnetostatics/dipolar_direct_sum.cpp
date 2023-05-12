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
/**
 * @brief Pair force of two interacting dipoles.
 *
 * @param d Distance vector.
 * @param m1 Dipole moment of one particle.
 * @param m2 Dipole moment of the other particle.
 *
 * @return Resulting force.
 */
auto pair_force(Utils::Vector3d const &d, Utils::Vector3d const &m1,
                Utils::Vector3d const &m2) {
  auto const pe2 = m1 * d;
  auto const pe3 = m2 * d;

  auto const r2 = d.norm2();
  auto const r = std::sqrt(r2);
  auto const r5 = r2 * r2 * r;
  auto const r7 = r5 * r2;

  auto const a = 3.0 * (m1 * m2) / r5;
  auto const b = -15.0 * pe2 * pe3 / r7;

  auto const f = (a + b) * d + 3.0 * (pe3 * m1 + pe2 * m2) / r5;
  auto const r3 = r2 * r;
  auto const t =
      -vector_product(m1, m2) / r3 + 3.0 * pe3 * vector_product(m1, d) / r5;

  return ParticleForce{f, t};
}

/**
 * @brief Pair potential for two interacting dipoles.
 *
 * @param d Distance vector.
 * @param m1 Dipole moment of one particle.
 * @param m2 Dipole moment of the other particle.
 *
 * @return Interaction energy.
 */
auto pair_potential(Utils::Vector3d const &d, Utils::Vector3d const &m1,
                    Utils::Vector3d const &m2) {
  auto const r2 = d * d;
  auto const r = sqrt(r2);
  auto const r3 = r2 * r;
  auto const r5 = r3 * r2;

  auto const pe1 = m1 * m2;
  auto const pe2 = m1 * d;
  auto const pe3 = m2 * d;

  return pe1 / r3 - 3.0 * pe2 * pe3 / r5;
}

/**
 * @brief Dipole field of a particle with dip_mom m1 @ a distance d.
 *
 * @param d Distance vector.
 * @param m1 Dipole moment of one particle.
 *
 * @return Utils::Vector3d wcontaining dipole field components.
 */
auto dipole_field(Utils::Vector3d const &d, Utils::Vector3d const &m1) {
  auto const r2 = d * d;
  auto const r = sqrt(r2);
  auto const r3 = r2 * r;
  auto const r5 = r3 * r2;
  auto const pe2 = m1 * d;

  return 3.0 * pe2 * d / r5 - m1 / r3;
}

/**
 * @brief Call kernel for every 3d index in a sphere around the origin.
 *
 * This calls a callable for all index-triples
 * that are within ball around the origin with
 * radius |ncut|.
 *
 * @tparam F Callable
 * @param ncut Limits in the three directions,
 *             all non-zero elements have to be
 *             the same number.
 * @param f will be called for each index triple
 *        within the limits of @p ncut.
 */
template <typename F> void for_each_image(Utils::Vector3i const &ncut, F f) {
  auto const ncut2 = ncut.norm2();

  /* This runs over the index "cube"
   * [-ncut[0], ncut[0]] x ... x [-ncut[2], ncut[2]]
   * (inclusive on both sides), and calls f with
   * all the elements as argument. Counting range
   * is a range that just enumerates a range.
   */
  Utils::cartesian_product(
      [&](int nx, int ny, int nz) {
        if (nx * nx + ny * ny + nz * nz <= ncut2) {
          f(nx, ny, nz);
        }
      },
      boost::counting_range(-ncut[0], ncut[0] + 1),
      boost::counting_range(-ncut[1], ncut[1] + 1),
      boost::counting_range(-ncut[2], ncut[2] + 1));
}

/**
 * @brief Position and dipole moment of one particle.
 */
struct PosMom {
  Utils::Vector3d pos;
  Utils::Vector3d m;

  template <class Archive> void serialize(Archive &ar, long int) { ar &pos &m; }
};

/**
 * @brief Sum over all pairs with periodic images.
 *
 * This implements the "primed" pair sum, the sum over all
 * pairs between one particles and all other particles,
 * including @p ncut periodic replicas in each direction.
 * Primed means that in the primary replica the self-interaction
 * is excluded, but not with the other periodic replicas. E.g.
 * a particle does not interact with its self, but does with
 * its periodically shifted versions.
 *
 * @param begin Iterator pointing to begin of particle range
 * @param end Iterator pointing past the end of particle range
 * @param it Pointer to particle that is considered
 * @param with_replicas If periodic replicas are to be considered
 *        at all. If false, distances are calulated as Euclidean
 *        distances, and not using minimum image convention.
 * @param ncut Number of replicas in each direction.
 * @param box_l Box dimensions.
 * @param init Initial value of the sum.
 * @param f Binary operation mapping distance and moment of the
 *          interaction partner to the value to be summed up for this pair.
 *
 * @return The total sum.
 */
template <class InputIterator, class T, class F>
T image_sum(InputIterator begin, InputIterator end, InputIterator it,
            bool with_replicas, Utils::Vector3i const &ncut,
            Utils::Vector3d const &box_l, T init, F f) {

  for (auto jt = begin; jt != end; ++jt) {
    auto const exclude_primary = (it == jt);
    auto const primary_distance =
        (with_replicas) ? (it->pos - jt->pos)
                        : ::box_geo.get_mi_vector(it->pos, jt->pos);

    for_each_image(ncut, [&](int nx, int ny, int nz) {
      if (!(exclude_primary && nx == 0 && ny == 0 && nz == 0)) {
        auto const rn =
            primary_distance +
            Utils::Vector3d{nx * box_l[0], ny * box_l[1], nz * box_l[2]};
        init += f(rn, jt->m);
      }
    });
  }

  return init;
}

auto gather_particle_data(ParticleRange const &particles, int n_replicas) {
  auto const &comm = ::comm_cart;
  std::vector<Particle *> local_particles;
  std::vector<PosMom> local_posmom;
  std::vector<PosMom> all_posmom;
  std::vector<boost::mpi::request> reqs;

  local_particles.reserve(particles.size());
  local_posmom.reserve(particles.size());

  for (auto &p : particles) {
    if (p.dipm() != 0.0) {
      local_particles.emplace_back(&p);
      local_posmom.emplace_back(
          PosMom{folded_position(p.pos(), ::box_geo), p.calc_dip()});
    }
  }

  auto const local_size = static_cast<int>(local_posmom.size());
  std::vector<int> all_sizes;
  boost::mpi::all_gather(comm, local_size, all_sizes);

  auto const offset =
      std::accumulate(all_sizes.begin(), all_sizes.begin() + comm.rank(), 0);
  auto const total_size =
      std::accumulate(all_sizes.begin() + comm.rank(), all_sizes.end(), offset);

  if (comm.size() > 1) {
    all_posmom.resize(total_size);
    reqs = Utils::Mpi::iall_gatherv(comm, local_posmom.data(), local_size,
                                    all_posmom.data(), all_sizes.data());
  } else {
    std::swap(all_posmom, local_posmom);
  }

  return std::make_tuple(std::move(local_particles), std::move(all_posmom),
                         std::move(reqs), offset);
}

auto get_n_cut(int n_replicas) {
  return n_replicas * Utils::Vector3i{static_cast<int>(::box_geo.periodic(0)),
                                      static_cast<int>(::box_geo.periodic(1)),
                                      static_cast<int>(::box_geo.periodic(2))};
}

} // namespace

/**
 * @brief Calculate and add the interaction forces/torques to the particles.
 *
 * This employs a parallel N-square loop over all particle pairs.
 * The computation the partitioned into several steps so that the
 * communication latency can be hidden behinder some local computation:
 *
 * 1. The local particle positions and momenta are packed into
 *    one array.
 * 2. The asynchronous distribution of the local arrays to all
 *    ranks is started.
 * 3. The interaction for the local pairs is started, here every
 *    pair is visited only once, and the force is added to both particles.
 * 4. Wait for the data from the other nodes.
 * 5. Calculate the interaction with the rest of the particles. Here
 *    every pair is visited twice (not necessarily on the same rank)
 *    so that no reduction of the forces is needed.
 *
 * Logically this is equivalent to the potential calculation
 * in @ref DipolarDirectSum::long_range_energy, which calculates
 * a naive N-square sum, but has better performance and scaling.
 */
void DipolarDirectSum::add_long_range_forces(
    ParticleRange const &particles) const {
  auto const &box_l = ::box_geo.length();
  auto [local_particles, all_posmom, reqs, offset] =
      gather_particle_data(particles, n_replicas);

  /* Number of image boxes considered */
  auto const ncut = get_n_cut(n_replicas);
  auto const with_replicas = (ncut.norm2() > 0);

  /* Range of particles we calculate the ia for on this node */
  auto const local_posmom_begin = all_posmom.begin() + offset;
  auto const local_posmom_end =
      local_posmom_begin + static_cast<long>(local_particles.size());

  /* Output iterator for the force */
  auto p = local_particles.begin();

  /* IA with local particles */
  for (auto it = local_posmom_begin; it != local_posmom_end; ++it, ++p) {
    /* IA with own images */
    auto fi = image_sum(
        it, std::next(it), it, with_replicas, ncut, box_l, ParticleForce{},
        [it](Utils::Vector3d const &rn, Utils::Vector3d const &mj) {
          return pair_force(rn, it->m, mj);
        });

    /* IA with other local particles */
    auto q = std::next(p);
    for (auto jt = std::next(it); jt != local_posmom_end; ++jt, ++q) {
      auto const d = (with_replicas)
                         ? (it->pos - jt->pos)
                         : ::box_geo.get_mi_vector(it->pos, jt->pos);

      ParticleForce fij{};
      ParticleForce fji{};
      for_each_image(ncut, [&](int nx, int ny, int nz) {
        auto const rn =
            d + Utils::Vector3d{nx * box_l[0], ny * box_l[1], nz * box_l[2]};
        auto const pf = pair_force(rn, it->m, jt->m);
        fij += pf;
        fji.f -= pf.f;
        /* Conservation of angular momentum mandates that
         * 0 = t_i + r_ij x F_ij + t_j */
        fji.torque += vector_product(pf.f, rn) - pf.torque;
      });

      fi += fij;
      (*q)->f.f += prefactor * fji.f;
      (*q)->f.torque += prefactor * fji.torque;
    }

    (*p)->force() += prefactor * fi.f;
    (*p)->torque() += prefactor * fi.torque;
  }

  /* Wait for the rest of the data to arrive */
  boost::mpi::wait_all(reqs.begin(), reqs.end());

  /* Output iterator for the force */
  p = local_particles.begin();

  /* Interaction with all the other particles */
  for (auto it = local_posmom_begin; it != local_posmom_end; ++it, ++p) {
    // red particles
    auto fi =
        image_sum(all_posmom.begin(), local_posmom_begin, it, with_replicas,
                  ncut, box_l, ParticleForce{},
                  [it](Utils::Vector3d const &rn, Utils::Vector3d const &mj) {
                    return pair_force(rn, it->m, mj);
                  });

    // black particles
    fi += image_sum(local_posmom_end, all_posmom.end(), it, with_replicas, ncut,
                    box_l, ParticleForce{},
                    [it](Utils::Vector3d const &rn, Utils::Vector3d const &mj) {
                      return pair_force(rn, it->m, mj);
                    });

    (*p)->force() += prefactor * fi.f;
    (*p)->torque() += prefactor * fi.torque;
  }
}

/**
 * @brief Calculate the interaction potential.
 *
 * This employs a parallel N-square loop over all particle pairs.
 */
double
DipolarDirectSum::long_range_energy(ParticleRange const &particles) const {
  auto const &box_l = ::box_geo.length();
  auto [local_particles, all_posmom, reqs, offset] =
      gather_particle_data(particles, n_replicas);

  /* Number of image boxes considered */
  auto const ncut = get_n_cut(n_replicas);
  auto const with_replicas = (ncut.norm2() > 0);

  /* Wait for the rest of the data to arrive */
  boost::mpi::wait_all(reqs.begin(), reqs.end());

  /* Range of particles we calculate the ia for on this node */
  auto const local_posmom_begin = all_posmom.begin() + offset;
  auto const local_posmom_end =
      local_posmom_begin + static_cast<long>(local_particles.size());

  auto u = 0.;
  for (auto it = local_posmom_begin; it != local_posmom_end; ++it) {
    u = image_sum(it, all_posmom.end(), it, with_replicas, ncut, box_l, u,
                  [it](Utils::Vector3d const &rn, Utils::Vector3d const &mj) {
                    return pair_potential(rn, it->m, mj);
                  });
  }

  return prefactor * u;
}
/**
 * @brief Calculate total dipole field at the position of each particle and
 * store it in a part property called dip_fld.
 *
 * This employs a parallel N-square loop over all particles.
 * Logically this is equivalent to the potential calculation
 * in @ref DipolarDirectSum::long_range_energy, which calculates
 * a naive N-square sum. The difference is summation range and kernel calcluated
 * the dipole filed rather than the energy. Threfore the return in of type
 * Vector3D.
 */
void DipolarDirectSum::dipole_field_at_part(
    ParticleRange const &particles) const {
  auto const &box_l = ::box_geo.length();
  /* collect particle data */
  auto [local_particles, all_posmom, reqs, offset] =
      gather_particle_data(particles, n_replicas);

  auto const ncut = get_n_cut(n_replicas);
  auto const with_replicas = (ncut.norm2() > 0);

  boost::mpi::wait_all(reqs.begin(), reqs.end());

  auto const local_posmom_begin = all_posmom.begin() + offset;
  auto const local_posmom_end =
      local_posmom_begin + static_cast<long>(local_particles.size());

  Utils::Vector3d u_init = {0., 0., 0.};
  auto p = local_particles.begin();
  for (auto pi = local_posmom_begin; pi != local_posmom_end; ++pi, ++p) {
    auto u = image_sum(
        all_posmom.begin(), all_posmom.end(), pi, with_replicas, ncut, box_l,
        u_init, [](Utils::Vector3d const &rn, Utils::Vector3d const &mj) {
          return dipole_field(rn, mj);
        });
    (*p)->dip_fld() = prefactor * u;
  }
}

double phi_objective(unsigned n, const double *x, double *grad,
                     void *my_func_data) {
  double phi = x[0];
  double *params = (double *)my_func_data;
  double theta = params[0];
  double h = params[1];
  if (grad) {
    grad[0] = 0.5 * sin(2 * (phi - theta)) + h * sin(phi);
  }
  return 0.25 - 0.25 * cos(2 * (phi - theta)) - h * cos(phi);
}

double inv_phi_objective(unsigned n, const double *x, double *grad,
                         void *my_func_data) {
  double phi = x[0];
  double *params = (double *)my_func_data;
  double theta = params[0];
  double h = params[1];
  if (grad) {
    grad[0] = -0.5 * sin(2 * (phi - theta)) - h * sin(phi);
  }
  return -0.25 + 0.25 * cos(2 * (phi - theta)) + h * cos(phi);
}
/*
SW energy minimisation step with kinetic MC step. SW energy normalised by the
anisotropy field H_k (Hkinv stored on part to avoid division, h is the reduced
field  due to the normalisation)


*/
double DipolarDirectSum::funct(double theta, double h, double phi0,
                               double kT_KVm_inv, double tau0_inv,
                               double dt) const {
  std::random_device rd;
  std::default_random_engine generator(rd());
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  double eps_phi = 1e-3;

  nlopt::opt opt(nlopt::LD_LBFGS, 1);
  double params[] = {theta, h};
  opt.set_min_objective(phi_objective, &params);

  /* relative tolerance should in principle be order k_bT. Minimise U_{SW} from
   * both sides. Possible one global minimum of two local minima. Using
   * constrained optimisation by quadratic approximations. See
   * https://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf
   */

  opt.set_ftol_abs(1e-8);
  opt.set_lower_bounds(phi0 - M_PI);
  opt.set_upper_bounds(phi0 + M_PI);
  std::vector<double> x(1);
  x[0] = phi0 + eps_phi; /* make initial guess from previos position plus
                                   an arbitrary perturbation*/
  double min1; /* this is the actuall value of the energy from minimiser */
  opt.optimize(x, min1);
  double phi_min1 = x[0];
  x[0] = phi0 + eps_phi -
         M_PI; /*try to find another minimimum from the other side*/
  double min2;
  opt.optimize(x, min2);
  double phi_min2 = x[0];

  if (phi_min2 < -M_PI) {
    phi_min2 += TWO_M_PI;
  } else if (phi_min2 > M_PI) {
    phi_min2 -= TWO_M_PI;
  }
  double sol;

  /* If there more that one minimum in the U_{SW} run kinetic MC step. Find
   * U_{SW} maxima to calculate the barried height for the jump probabilities.
   * Same logic as before, same issues */

  if (fabs(phi_min1 - phi_min2) > 1.e-3) {
    opt.set_min_objective(inv_phi_objective, &params);
    x[0] = phi0 + eps_phi;
    double max1;
    opt.optimize(x, max1);
    double phi_max1 = x[0];

    x[0] = phi0 + eps_phi - M_PI;
    double max2;
    opt.optimize(x, max2);
    double phi_max2 = x[0];

    if (phi_max1 < -M_PI) {
      phi_max1 += TWO_M_PI;
    } else if (phi_max1 > M_PI) {
      phi_max1 -= TWO_M_PI;
    }
    if (phi_max2 < -M_PI) {
      phi_max2 += TWO_M_PI;
    } else if (phi_max2 > M_PI) {
      phi_max2 -= TWO_M_PI;
    }

    double b1 = (phi_objective(1, &phi_max1, nullptr, &params) -
                 phi_objective(1, &phi_min1, nullptr, &params)) *
                kT_KVm_inv;
    double b2 = (phi_objective(1, &phi_max2, nullptr, &params) -
                 phi_objective(1, &phi_min2, nullptr, &params)) *
                kT_KVm_inv;
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
  return sol;
}

void DipolarDirectSum::stoner_wolfarth_main(
    ParticleRange const &particles) const {
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
      double theta = acos(e_h * e_k);
      auto rot_axis =
          vector_product(vector_product(e_h, e_k), e_h).normalized();
      if (theta > M_PI_2) {
        theta = M_PI - theta;
        h = -h;
        e_h = -e_h;
      }
      auto phi = fmod(funct(theta, h, (*pi)->phi0(), (*pi)->kT_KVm_inv(),
                            (*pi)->tau0_inv(), (*pi)->dt_incr()),
                      TWO_M_PI);
      (*pi)->phi0() = phi;
      auto mom = e_h * cos(phi) + rot_axis * sin(phi);
      auto const [quat, dipm] = convert_dip_to_quat((*p)->sat_mag() * mom);
      (*p)->dipm() = dipm;
      (*p)->quat() = quat;
    }
    on_dipoles_change();
  } else {
    runtimeWarningMsg() << "we are in else";

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

DipolarDirectSum::DipolarDirectSum(double prefactor, int n_replicas)
    : prefactor{prefactor}, n_replicas{n_replicas} {
  if (prefactor < 0.) {
    throw std::domain_error("Parameter 'prefactor' must be >= 0");
  }
  if (n_replicas < 0) {
    throw std::domain_error("Parameter 'n_replicas' must be >= 0");
  }
}

#endif // DIPOLES
