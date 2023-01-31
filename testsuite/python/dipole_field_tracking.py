import espressomd
import espressomd.magnetostatics
import numpy as np
import unittest as ut
import unittest_decorators as utx
import tests_common


def dip_fld_kernel(dipx, dipy, dipz, mx, my, mz, x, y, z):
    dx, dy, dz = x-dipx, y-dipy, z-dipz
    dr = np.linalg.norm((dx, dy, dz))
    dr3, dr5 = pow(dr, 3.0), pow(dr, 5.0)
    mr = dx*mx+dy*my+dz*mz
    Hx, Hy, Hz = 3.*dx*mr/dr5-mx/dr3, 3.*dy*mr/dr5-my/dr3, 3.*dz*mr/dr5-mz/dr3
    return Hx, Hy, Hz


def N2_loop(particle, slice_prop):
    storage = [0, 0, 0]
    particle_index, particle_x, particle_y, particle_z = particle.id, *particle.pos
    for part_id, part_pos, part_dip in slice_prop:
        if part_id != particle_index:
            dip_x, dip_y, dip_z = part_pos
            m_dip_x, m_dip_y, m_dip_z = part_dip
            dipole_field = dip_fld_kernel(
                dip_x, dip_y, dip_z, m_dip_x, m_dip_y, m_dip_z, particle_x, particle_y, particle_z)
            storage[0] = storage[0] + dipole_field[0]
            storage[1] = storage[1] + dipole_field[1]
            storage[2] = storage[2] + dipole_field[2]
    return storage


def compare_dipole_fields(p_slice):
    slice_data = [(x.id, x.pos, x.dip) for x in p_slice]
    ret_bool_list = [np.allclose(p.dip_fld, N2_loop(p, slice_data)) for p in p_slice]
    return ret_bool_list


@utx.skipIfMissingFeatures(["DIPOLE_FIELD_TRACKING"])
class DipoleFieldsLJFluid(ut.TestCase):
    '''
    ut.TestCase subclass testing the total dipole field for a frozen LJ fluid @ every particle. By hand calculation compared with espresso DIPOLE_FIELD_TRACKING values.
    '''
    system = espressomd.System(box_l=[1.0, 1.0, 1.0])
    data = np.loadtxt(tests_common.data_path('lj_system.dat'))
    pos = data[:, 1:4]

    def tearDown(self):
        self.system.part.clear()
        self.system.actors.clear()
        self.system.periodicity = [False, False, False]

    def setUp(self):
        self.system.part.clear()
        self.system.box_l = [10.7437] * 3
        self.system.periodicity = [False, False, False]

        self.system.cell_system.skin = 0.4
        self.system.time_step = .1

        self.system.part.add(pos=self.pos)
        solver = espressomd.magnetostatics.DipolarDirectSumCpu(prefactor=1.)
        self.system.actors.add(solver)
        self.system.part.all().dip = (1, 0, 0)

    def test_dd(self):

        self.system.integrator.run(steps=0)
        all_dipfld_bool = compare_dipole_fields(self.system.part.all())
        assert all(all_dipfld_bool)


if __name__ == '__main__':
    ut.main()
