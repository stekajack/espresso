# Copyright (C) 2010-2019 The ESPResSo project
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
from .script_interface import ScriptObjectRegistry, ScriptInterfaceHelper, script_interface_register
from .utils import check_type_or_throw_except
from .__init__ import has_features
import numpy as np


if has_features(["LB_BOUNDARIES"]):
    @script_interface_register
    class LBBoundaries(ScriptObjectRegistry):

        """
        Creates a set of lattice-Boltzmann boundaries.

        """

        _so_name = "LBBoundaries::LBBoundaries"

        def add(self, *args, **kwargs):
            """
            Adds a boundary to the set of boundaries.
            Either pass a valid boundary as argument,
            or a valid set of parameters to create a boundary.

            """

            if len(args) == 1:
                if isinstance(args[0], LBBoundary):
                    lbboundary = args[0]
                else:
                    raise TypeError(
                        "Either a LBBoundary object or key-value pairs for the parameters of a LBBoundary object need to be passed.")
            else:
                lbboundary = LBBoundary(**kwargs)
            self.call_method("add", object=lbboundary)
            return lbboundary

        def remove(self, lbboundary):
            """
            Removes a boundary from the set.

            Parameters
            ----------
            lbboundary : :obj:`LBBoundary`
                The boundary to be removed from the set.

            """

            self.call_method("remove", object=lbboundary)

        def clear(self):
            """
            Removes all boundaries.

            """

            self.call_method("clear")

        def size(self):
            return self.call_method("size")

        def empty(self):

            return self.call_method("empty")

    @script_interface_register
    class LBBoundary(ScriptInterfaceHelper):

        """
        Creates a LB boundary from a shape.

        The fluid velocity is limited to :math:`v_{\\mathrm{max}} = 0.20`
        (see *quasi-incompressible limit* in :cite:`kruger17a`,
        chapter 7, page 272), which corresponds to Mach 0.35.

        The relative error in the fluid density between a compressible fluid
        and an incompressible fluid at Mach 0.30 is less than 5% (see
        *constant density assumption* in :cite:`kundu01a` chapter 16, page
        663). Since the speed of sound is :math:`c_s = 1 / \\sqrt{3}` in LB
        velocity units in a D3Q19 lattice, the velocity limit at Mach 0.30
        is :math:`v_{\\mathrm{max}} = 0.30 / \\sqrt{3} \\approx 0.17`.
        At Mach 0.35 the relative error is around 6% and
        :math:`v_{\\mathrm{max}} = 0.35 / \\sqrt{3} \\approx 0.20`.

        Parameters
        ----------
        shape : :obj:`espressomd.shapes.Shape`
            The shape from which to build the boundary.
        velocity : (3,) array_like of :obj:`float`, optional
            The boundary slip velocity. By default, a velocity of zero is used
            (no-slip boundary condition).

        """

        _so_name = "LBBoundaries::LBBoundary"
        _so_bind_methods = ("get_force",)


def edge_detection(boundary_mask, periodicity):
    """
    Find boundary nodes in contact with the fluid. Relies on a convolution
    kernel constructed from the D3Q19 stencil.

    Parameters
    ----------
    boundary_mask : (N, M, L) array_like of :obj:`bool`
        Bitmask for the rasterized boundary geometry.
    periodicity : (3,) array_like of :obj:`bool`
        Bitmask for the box periodicity.

    Returns
    -------
    (N, 3) array_like of :obj:`int`
        The indices of the boundary nodes at the interface with the fluid.
    """
    import scipy.signal
    import itertools

    fluid_mask = np.logical_not(boundary_mask)

    # edge kernel
    edge = -np.ones((3, 3, 3))
    for i, j, k in itertools.product((0, 2), (0, 2), (0, 2)):
        edge[i, j, k] = 0
    edge[1, 1, 1] = -np.sum(edge)

    # periodic convolution
    wrapped_mask = np.pad(fluid_mask.astype(int), 3 * [(2, 2)], mode='wrap')
    if not periodicity[0]:
        wrapped_mask[:2, :, :] = 0
        wrapped_mask[-2:, :, :] = 0
    if not periodicity[1]:
        wrapped_mask[:, :2, :] = 0
        wrapped_mask[:, -2:, :] = 0
    if not periodicity[2]:
        wrapped_mask[:, :, :2] = 0
        wrapped_mask[:, :, -2:] = 0
    convolution = scipy.signal.convolve(
        wrapped_mask, edge, mode='same', method='direct')[2:-2, 2:-2, 2:-2]
    convolution = np.multiply(convolution, boundary_mask)

    return np.array(np.nonzero(convolution < 0)).T


def calc_cylinder_tangential_vectors(center, agrid, offset, node_indices):
    """
    Utility function to calculate a constant slip velocity tangential to the
    surface of a cylinder.

    Parameters
    ----------
    center : (3,) array_like of :obj:`float`
        Center of the cylinder.
    agrid : :obj:`float`
        LB agrid.
    offset : :obj:`float`
        LB offset.
    node_indices : (N, 3) array_like of :obj:`int`
        Indices of the boundary surface nodes.

    Returns
    -------
    (N, 3) array_like of :obj:`float`
        The unit vectors tangential to the surface of a cylinder.
    """
    velocities = []
    for ijk in node_indices:
        p = (ijk + offset) * agrid
        r = center - p
        norm = np.linalg.norm(r[:2])
        if norm < 1e-10:
            velocities.append(np.zeros(3))
            continue
        angle_r = np.arccos(np.dot(r[:2] / norm, [1, 0]))
        angle_v = angle_r - np.pi / 2
        flip = np.sign(r[1])
        slip_velocity = np.array([flip * np.cos(angle_v), np.sin(angle_v), 0.])
        velocities.append(slip_velocity)
    return np.array(velocities)
