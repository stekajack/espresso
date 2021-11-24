#
# Copyright (C) 2013-2019 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import numpy as np

import espressomd
import espressomd.shapes
import espressomd.virtual_sites
import espressomd.utils

import tests_common
import unittest_decorators as utx


class VirtualSitesTracersCommon:
    box_height = 10.
    box_lw = 8.
    system = espressomd.System(box_l=(box_lw, box_lw, box_height))
    system.time_step = 0.08
    system.cell_system.skin = 0.1

    def tearDown(self):
        self.system.part.clear()
        self.system.actors.clear()
        self.system.thermostat.turn_off()

    def reset_lb(self, ext_force_density=(0, 0, 0)):
        self.lbf = self.LBClass(
            kT=0.0, agrid=1, density=1, viscosity=1.8,
            tau=self.system.time_step, ext_force_density=ext_force_density)
        self.system.actors.add(self.lbf)
        self.system.thermostat.set_lb(
            LB_fluid=self.lbf,
            act_on_virtual=False,
            gamma=1)

        # Setup boundaries
        wall_shapes = [None] * 2
        wall_shapes[0] = espressomd.shapes.Wall(normal=[0, 0, 1], dist=0.5)
        wall_shapes[1] = espressomd.shapes.Wall(
            normal=[0, 0, -1], dist=-self.box_height - 0.5)

        for wall_shape in wall_shapes:
            self.lbf.add_boundary_from_shape(wall_shape)

        espressomd.utils.handle_errors("setup")

    def test_aa_method_switching(self):
        # Virtual sites should be disabled by default
        self.assertIsInstance(
            self.system.virtual_sites,
            espressomd.virtual_sites.VirtualSitesOff)

        # Switch implementation
        self.system.virtual_sites = espressomd.virtual_sites.VirtualSitesInertialessTracers()
        self.assertIsInstance(
            self.system.virtual_sites, espressomd.virtual_sites.VirtualSitesInertialessTracers)

    @utx.skipIfMissingFeatures("EXTERNAL_FORCES")
    def test_ab_single_step(self):
        self.reset_lb()
        self.lbf.clear_boundaries()
        self.system.part.clear()
        self.system.virtual_sites = espressomd.virtual_sites.VirtualSitesInertialessTracers()

        # Random velocities
        for n in self.lbf.nodes():
            n.velocity = np.random.random(3) - .5
        force = [1, -2, 3]
        # Test several particle positions
        for pos in [[3, 2, 1], [0, 0, 0],
                    self.system.box_l * 0.49,
                    self.system.box_l,
                    self.system.box_l * 0.99]:
            p = self.system.part.add(pos=pos, ext_force=force, virtual=True)

            coupling_pos = p.pos
            # Nodes to which forces will be interpolated
            lb_nodes = tests_common.get_lb_nodes_around_pos(
                coupling_pos, self.lbf)

            np.testing.assert_allclose(
                [n.last_applied_force for n in lb_nodes],
                np.zeros((len(lb_nodes), 3)))
            self.system.integrator.run(1)

            v_fluid = np.copy(self.lbf.get_interpolated_velocity(coupling_pos))

            # Check particle velocity
            np.testing.assert_allclose(np.copy(p.v), v_fluid)

            # particle position
            np.testing.assert_allclose(
                np.copy(p.pos),
                coupling_pos + v_fluid * self.system.time_step)

            # check transfer of particle force to fluid
            applied_forces = np.array([n.last_applied_force for n in lb_nodes])
            np.testing.assert_allclose(
                np.sum(applied_forces, axis=0), force, atol=1E-10)

            # Check that last_applied_force gets cleared
            p.remove()
            self.system.integrator.run(1)
            applied_forces = np.array([n.last_applied_force for n in lb_nodes])
            np.testing.assert_allclose(
                np.sum(applied_forces, axis=0), [0, 0, 0])

    def test_advection(self):
        self.reset_lb(ext_force_density=[0.1, 0, 0])
        # System setup
        system = self.system

        system.virtual_sites = espressomd.virtual_sites.VirtualSitesInertialessTracers()

        # Establish steady state flow field
        p = system.part.add(pos=(0, 5.5, 5.5), virtual=True)
        system.integrator.run(400)

        p.pos = (0, 5.5, 5.5)
        system.time = 0

        # Perform integration
        for _ in range(2):
            system.integrator.run(100)
            # compute expected position
            dist = self.lbf.get_interpolated_velocity(p.pos)[0] * system.time
            self.assertAlmostEqual(p.pos[0] / dist, 1, delta=0.001)

    def test_zz_without_lb(self):
        """Check behaviour without lb. Ignore non-virtual particles, complain on
        virtual ones.

        """
        self.reset_lb()
        system = self.system
        system.virtual_sites = espressomd.virtual_sites.VirtualSitesInertialessTracers()
        system.actors.clear()
        system.part.clear()
        p = system.part.add(pos=(0, 0, 0))
        system.integrator.run(1)
        p.virtual = True
        with self.assertRaises(Exception):
            system.integrator.run(1)
