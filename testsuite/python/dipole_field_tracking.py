import espressomd
import espressomd.magnetostatics
import numpy as np
import unittest as ut
import unittest_decorators as utx
import tests_common


def dip_fld_kernel(dipx, dipy, dipz, mx, my, mz, x, y, z):
    dx, dy, dz = x - dipx, y - dipy, z - dipz
    dr = np.linalg.norm((dx, dy, dz))
    dr3, dr5 = pow(dr, 3.0), pow(dr, 5.0)
    mr = dx * mx + dy * my + dz * mz
    Hx, Hy, Hz = 3. * dx * mr / dr5 - mx / dr3, 3. * dy * \
        mr / dr5 - my / dr3, 3. * dz * mr / dr5 - mz / dr3
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


@utx.skipIfMissingFeatures(["DIPSUS"])
class DipoleFieldsLJFluid(ut.TestCase):
    '''
    ut.TestCase subclass testing the total dipole field for a LJ fluid ( 500 particles, density approx 0.002, mu^2=1, no PBC) @ every particle. By hand calculation compared with espresso DIPOLE_FIELD_TRACKING values.
    '''
    system = espressomd.System(box_l=[1.0, 1.0, 1.0])
    data = np.loadtxt(tests_common.data_path('lj_system.dat'))
    pos = data[:, 1:4]

    def tearDown(self):
        self.system.part.clear()
        self.system.actors.clear()

    def setUp(self):
        np.random.seed(42)
        self.system.part.clear()
        self.system.box_l = (50, 50, 50)

        self.system.cell_system.skin = 0.4
        self.system.time_step = .1

        self.system.part.add(type=[0] * 500, pos=np.random.random((500, 3))
                             * self.system.box_l)
        self.system.part.all().rotation = (True, True, True)


        # minimize system
        self.system.non_bonded_inter[0, 0].lennard_jones.set_params(
            epsilon=10.0, sigma=1, cutoff=2**(1.0 / 6.0), shift="auto")
        self.system.integrator.set_steepest_descent(
            f_max=1, gamma=0.001, max_displacement=0.01)
        self.system.integrator.run(100)
        self.system.non_bonded_inter[0, 0].lennard_jones.deactivate()
        self.system.integrator.set_vv()
        assert self.system.analysis.energy()["total"] == 0

        orientor_list = [np.reshape(np.random.randn(3, 1), (3, ))
                         for x in range(500)]
        orientor_list_normalized = np.array(
            [(x/np.linalg.norm(x, axis=0)) for x in orientor_list])
        dip_mom = orientor_list_normalized
        self.system.part.all().dip = dip_mom

        # solver = espressomd.magnetostatics.DipolarDirectSumCpu(prefactor=1.)

    @staticmethod
    def rotation_matrix(quat):
        """
        Converts a quaternion to a rotation matrix.
        
        Parameters:
            quat (np.ndarray): A quaternion as a 4-element numpy array [w, x, y, z].
            
        Returns:
            np.ndarray: A 3x3 rotation matrix.
        """
        w, x, y, z = quat
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])
    
    @staticmethod
    def convert_vector_space_to_body(p, v):
        """
        Converts a vector from world space to body-fixed space of a particle.
        
        Parameters:
            p (dict): A dictionary containing particle properties, must include 'quat'.
                    'quat' is expected to be a 4-element numpy array [w, x, y, z].
            v (np.ndarray): A 3D vector in world space.
            
        Returns:
            np.ndarray: The 3D vector in the particle's body-fixed coordinate space.
        """
        quat = p['quat']
        assert np.linalg.norm(quat) > 0.0, "Quaternion norm must be greater than zero."
        
        rot_matrix = DipoleFieldsLJFluid.rotation_matrix(quat).T  # Transpose of the rotation matrix
        return np.dot(rot_matrix, v)
    
    @staticmethod
    def convert_vector_space_to_world(p, v):
        """
        Converts a vector from body-fixed space to world space of a particle.
        
        Parameters:
            p (dict): A dictionary containing particle properties, must include 'quat'.
                    'quat' is expected to be a 4-element numpy array [w, x, y, z].
            v (np.ndarray): A 3D vector in body-fixed space.
            
        Returns:
            np.ndarray: The 3D vector in the world coordinate space.
        """
        quat = p['quat']
        assert np.linalg.norm(quat) > 0.0, "Quaternion norm must be greater than zero."
        
        rot_matrix = DipoleFieldsLJFluid.rotation_matrix(quat)  # Use the original rotation matrix
        return np.dot(rot_matrix, v)

    def test_dd(self):
        self.system.periodicity = [True, True, True]
        self.system.actors.clear()
        solver = espressomd.magnetostatics.DipolarP3M(prefactor=1, accuracy=1E-8,epsilon=1)
        self.system.actors.add(solver)
        self.system.integrator.run(steps=1)
        for p in self.system.part.all():
            np.testing.assert_allclose(np.copy(p.torque_lab), np.cross(p.dip,p.dip_fld))
    
    def test_ddas(self):
        self.system.periodicity = [True, True, True]
        self.system.actors.clear()
        solver = espressomd.magnetostatics.DipolarDirectSumCpu(prefactor=1.,n_replicas=2)
        self.system.actors.add(solver)
        self.system.integrator.run(steps=1)
        for p in self.system.part.all():
            np.testing.assert_allclose(np.copy(p.torque_lab), np.cross(p.dip,p.dip_fld))

if __name__ == '__main__':
    ut.main()