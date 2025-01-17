/*
 * Copyright (C) 2014-2019 The ESPResSo project
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
#ifndef ESPRESSO_CORE_ACTOR_MMM1DGPUFORCE_HPP
#define ESPRESSO_CORE_ACTOR_MMM1DGPUFORCE_HPP

#include "config.hpp"

#ifdef MMM1D_GPU

#include "Actor.hpp"
#include "SystemInterface.hpp"

class Mmm1dgpuForce : public Actor {
public:
  // constructor
  Mmm1dgpuForce(SystemInterface &s, float coulomb_prefactor, float maxPWerror,
                float far_switch_radius = -1, int bessel_cutoff = -1);
  ~Mmm1dgpuForce() override;
  // interface methods
  void computeForces(SystemInterface &s) override;
  void computeEnergy(SystemInterface &s) override;
  // configuration methods
  void setup(SystemInterface &s);
  void tune(SystemInterface &s, float _maxPWerror, float _far_switch_radius,
            int _bessel_cutoff);
  void set_params(float _boxz, float _coulomb_prefactor, float _maxPWerror,
                  float _far_switch_radius, int _bessel_cutoff,
                  bool manual = false);
  void activate();
  void deactivate();

private:
  // CUDA parameters
  unsigned int numThreads;
  unsigned int numBlocks(SystemInterface const &s) const;

  // the box length currently set on the GPU
  // Needed to make sure it hasn't been modified after inter coulomb was used.
  float host_boxz;
  // the number of particles we had during the last run. Needed to check if we
  // have to realloc dev_forcePairs
  unsigned int host_npart;
  bool need_tune;

  // pairs==0: return forces using atomicAdd
  // pairs==1: return force pairs
  // pairs==2: return forces using a global memory reduction
  int pairs;
  // variables for forces and energies calculated pre-reduction
  float *dev_forcePairs, *dev_energyBlocks;

  // MMM1D parameters
  float coulomb_prefactor, maxPWerror, far_switch_radius;
  int bessel_cutoff;

  // run a single force calculation and return the time it takes using
  // high-precision CUDA timers
  float force_benchmark(SystemInterface &s);

  // some functions to move MPI dependencies out of the .cu file
  void sanity_checks();
};

#endif
#endif
