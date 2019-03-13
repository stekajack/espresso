/*
  Copyright (C) 2010-2018 The ESPResSo project
  Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010
    Max-Planck-Institute for Polymer Research, Theory Group

  This file is part of ESPResSo.

  ESPResSo is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ESPResSo is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
/** \file
 * All 3d non P3M methods to deal with the
 * magnetic dipoles
 *
 *   MDDS => Calculates dipole-dipole interaction of a periodic system
 *   by explicitly summing the dipole-dipole interaction over several copies of
 * the system
 *   Uses spherical summation order
 *
 */

#include "electrostatics_magnetostatics/mdds.hpp"

#ifdef DIPOLES
#include "cells.hpp"
#include "grid.hpp"
#include "nonbonded_interactions/nonbonded_interaction_data.hpp"

#include "utils/cartesian_product.hpp"
#include "utils/for_each_pair.hpp"
#include "utils/mpi/all_gatherv.hpp"

#include <boost/mpi/collectives/all_gather.hpp>
#include <boost/range/algorithm_ext/for_each.hpp>
#include <boost/range/counting_range.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/range/numeric.hpp>
#include <boost/range/value_type.hpp>

/* =============================================================================
                  DIRECT SUM FOR MAGNETIC SYSTEMS
   =============================================================================
*/

int mdds_n_replicas = 0;

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
auto pair_force(Vector3d const &d, Vector3d const &m1, Vector3d const &m2)
    -> ParticleForce {
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
  auto const t = -m1.cross(m2) / r3 + 3.0 * pe3 * m1.cross(d) / r5;

  return {f, t};
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
auto pair_potential(Vector3d const &d, Vector3d const &m1, Vector3d const &m2)
    -> double {
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
 * @brief Call kernel for every 3-d index in
 *        a ball arcound the origin.
 *
 * This calls a Callable for all index-triples
 * that are within ball around the origin with
 * radius |ncut|.
 *
 * @tparam F Callable
 * @param ncut Limit in the three directions,
 *             all non-zero elements have to be
 *             the same number.
 * @param f will be called for each index triple
 *        within the limits of @p ncut.
 */
template <typename F> void for_each_image(Vector3i const &ncut, F f) {
  auto const ncut2 = ncut.norm2();

  using boost::counting_range;
  using Utils::cartesian_product;

  /* This runs over the index "cube"
   * [-ncut[0], ncut[0]] x ... x [-ncut[2], ncut[2]]
   * (inclusive on both sides), and calls f with
   *  all the elements as argument. Counting range
   *  is a range that just enumerates a range.
   */
  cartesian_product(
      [&](int nx, int ny, int nz) {
        if (nx * nx + ny * ny + nz * nz <= ncut2) {
          f(nx, ny, nz);
        }
      },
      counting_range(-ncut[0], ncut[0] + 1),
      counting_range(-ncut[1], ncut[1] + 1),
      counting_range(-ncut[2], ncut[2] + 1));
}

/**
 * @brief Position and moment of one particle.
 */
struct PosMom {
  Vector3d pos;
  Vector3d m;

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
 * @param pi Pointer to particle that is considered
 * @param with_replicas If periodic replicas are to be considered
 *        at all. If false, distences are calulated as Euclidian
 *        distances, and not using minimum image convention.
 * @param ncut Number of replicas in each direction.
 * @param init Initial value of the sum.
 * @param f Binary operation mapping distance and moment of the
 *          interaction partner to the value to be summed up for this pair.
 *
 * @return The total sum.
 */
template <class InputIterator, class T, class F>
T image_sum(InputIterator begin, InputIterator end, InputIterator pi,
            bool with_replicas, Vector3i const &ncut, T init, F f) {

  for (auto pj = begin; pj != end; ++pj) {
    auto const exclude_primary = (pi == pj);
    auto const primary_distance =
        (with_replicas) ? (pi->pos - pj->pos) : get_mi_vector(pi->pos, pj->pos);

    for_each_image(ncut, [&](int nx, int ny, int nz) {
      if (!(exclude_primary && nx == 0 && ny == 0 && nz == 0)) {
        auto const rn = primary_distance +
                        Vector3d{nx * box_l[0], ny * box_l[1], nz * box_l[2]};
        init += f(rn, pj->m);
      }
    });
  }

  return init;
}

template <typename T>
std::vector<T> all_gather(const boost::mpi::communicator &comm,
                          const T &local_value) {
  std::vector<T> all_values;
  boost::mpi::all_gather(comm, local_value, all_values);

  return all_values;
}

std::pair<int, int> offset_and_size(std::vector<int> const &sizes, int rank) {
  auto const offset = std::accumulate(sizes.begin(), sizes.begin() + rank, 0);
  auto const total_size =
      std::accumulate(sizes.begin() + rank, sizes.end(), offset);

  return {offset, total_size};
}

void collect_local_particles(const ParticleRange &particles,
                             std::vector<Particle *> &interacting_particles,
                             std::vector<PosMom> &posmom) {
  interacting_particles.reserve(interacting_particles.size() +
                                particles.size());
  posmom.reserve(posmom.size() + particles.size());

  for (auto &p : particles) {
    if (p.p.dipm != 0.0) {
      interacting_particles.emplace_back(&p);
      posmom.emplace_back(PosMom{folded_position(p.r.p), p.calc_dip()});
    }
  }
}

} // namespace

void mdds_forces(const ParticleRange &particles,
                 const boost::mpi::communicator &comm) {
  std::vector<Particle *> local_interacting_particles;
  std::vector<PosMom> local_posmom;

  collect_local_particles(particles, local_interacting_particles, local_posmom);

  auto const local_size = static_cast<int>(local_posmom.size());
  auto const sizes = all_gather(comm, local_size);

  int offset, total_size;
  std::tie(offset, total_size) = offset_and_size(sizes, comm.rank());

  std::vector<PosMom> all_posmom;
  std::vector<boost::mpi::request> reqs;
  if (comm.size() > 1) {
    all_posmom.resize(total_size);
    reqs = Utils::Mpi::iall_gatherv(comm, local_posmom.data(), local_size,
                                    all_posmom.data(), sizes.data());
  } else {
    std::swap(all_posmom, local_posmom);
  }

  /* Number of image boxes considered */
  const Vector3i ncut =
      mdds_n_replicas * Vector3i{static_cast<int>(PERIODIC(0)),
                                 static_cast<int>(PERIODIC(1)),
                                 static_cast<int>(PERIODIC(2))};
  auto const with_replicas = (ncut.norm2() > 0);

  /* Range of particles we calculate the ia for on this node */
  auto begin = all_posmom.begin() + offset;
  auto const end = begin + local_interacting_particles.size();

  /* Output iterator for the force */
  auto p = local_interacting_particles.begin();

  /* IA with local particles */
  for (auto pi = begin; pi != end; ++pi, ++p) {
    /* IA with own images */
    auto fi =
        image_sum(pi, std::next(pi), pi, with_replicas, ncut, ParticleForce{},
                  [pi](Vector3d const &rn, Vector3d const &mj) {
                    return pair_force(rn, pi->m, mj);
                  });

    auto q = std::next(p);
    for (auto pj = std::next(pi); pj != end; ++pj, ++q) {
      auto const d = (with_replicas) ? (pi->pos - pj->pos)
                                     : get_mi_vector(pi->pos, pj->pos);

      ParticleForce fij{};
      for_each_image(ncut, [&](int nx, int ny, int nz) {
        fij += pair_force(d, pi->m, pj->m);
      });

      fi += fij;
      (*q)->f.f -= coulomb.Dprefactor * fij.f;
      /* Conservation of angular momentum mandates that
       * 0 = t_i + r_ij x F_ij + t_j */
      (*q)->f.torque += coulomb.Dprefactor * (-fij.torque + fij.f.cross(d));
    }

    (*p)->f.f += coulomb.Dprefactor * fi.f;
    (*p)->f.torque += coulomb.Dprefactor * fi.torque;
  }

  boost::mpi::wait_all(reqs.begin(), reqs.end());

  p = local_interacting_particles.begin();
  for (auto pi = begin; pi != end; ++pi, ++p) {
    auto fi = image_sum(all_posmom.begin(), begin, pi, with_replicas, ncut,
                        ParticleForce{},
                        [pi](Vector3d const &rn, Vector3d const &mj) {
                          return pair_force(rn, pi->m, mj);
                        });

    fi += image_sum(end, all_posmom.end(), pi, with_replicas, ncut,
                    ParticleForce{},
                    [pi](Vector3d const &rn, Vector3d const &mj) {
                      return pair_force(rn, pi->m, mj);
                    });

    (*p)->f.f += coulomb.Dprefactor * fi.f;
    (*p)->f.torque += coulomb.Dprefactor * fi.torque;
  }
}

double mdds_energy(const ParticleRange &particles,
                   const boost::mpi::communicator &comm) {
  std::vector<Particle *> local_interacting_particles;
  std::vector<PosMom> local_posmom;

  collect_local_particles(particles, local_interacting_particles, local_posmom);

  const auto sizes = all_gather(comm, static_cast<int>(local_posmom.size()));

  int offset, total_size;
  std::tie(offset, total_size) = offset_and_size(sizes, comm.rank());

  std::vector<PosMom> all_posmom;
  if (comm.size() > 1) {
    all_posmom.resize(total_size);
    Utils::Mpi::all_gatherv(comm, local_posmom.data(), local_posmom.size(),
                            all_posmom.data(), sizes.data());
  } else {
    std::swap(all_posmom, local_posmom);
  }

  /* Number of image boxes considered */
  const Vector3i ncut =
      mdds_n_replicas * Vector3i{static_cast<int>(PERIODIC(0)),
                                 static_cast<int>(PERIODIC(1)),
                                 static_cast<int>(PERIODIC(2))};
  auto const with_replicas = (ncut.norm2() > 0);
  /* Range of particles we calculate the ia for on this node */
  auto begin = all_posmom.begin() + offset;
  auto const end = begin + local_interacting_particles.size();

  /* Output iterator for the force */
  auto p = local_interacting_particles.begin();

  auto u = 0.;
  for (auto pi = begin; pi != end; ++pi, ++p) {
    u = image_sum(pi, all_posmom.end(), pi, with_replicas, ncut, u,
                  [pi](Vector3d const &rn, Vector3d const &mj) {
                    return pair_potential(rn, pi->m, mj);
                  });
  }

  return coulomb.Dprefactor * u;
}

void mdds_set_params(int n_cut) {
  mdds_n_replicas = n_cut;

  if ((PERIODIC(0) || PERIODIC(1) || PERIODIC(2)) && mdds_n_replicas == 0) {
    fprintf(stderr, "Careful:  the number of extra replicas to take into "
                    "account during the direct sum calculation is zero \n");
  }

  if (coulomb.Dmethod != DIPOLAR_DS && coulomb.Dmethod != DIPOLAR_MDLC_DS) {
    set_dipolar_method_local(DIPOLAR_DS);
  }

  mpi_bcast_coulomb_params();
}

#endif