# cython: cdivision=True
from libc.math cimport sqrt, tanh
from .particle_data cimport ParticleHandle

cdef (double, double, double) calculate_generic_dipole_zfield_at(float dipx, float dipy, float dipz, float mx, float my, float mz, float x, float y, float z) nogil:
    cdef double dx
    cdef double dy
    cdef double dz
    cdef double dr
    cdef double dr3
    cdef double dr5
    cdef double mr
    cdef double Hx
    cdef double Hy
    cdef double Hz
    cdef (double, double, double) H_tot= (0,0,0)
    dx = x-dipx
    dy = y-dipy
    dz = z-dipz
    dr = sqrt(dx*dx+dy*dy+dz*dz)
    dr3 = dr**3
    dr5 = dr**5
    mr = dx*mx+dy*my+dz*mz
    H_tot[0] = 3*dx*mr/dr5-mx/dr3
    H_tot[1] = 3*dy*mr/dr5-my/dr3
    H_tot[2] = 3*dz*mr/dr5-mz/dr3
    return H_tot


cdef (float,float,float) effective_dipole_field_at_point((int,float,float,float) particle, list slice, dict neighbourhood_slice):
    cdef (float,float,float) storage = (0,0,0)
    particle_index, particle_x, particle_y, particle_z = particle
    for part_id, part_pos, part_dip in slice:
        if part_id in neighbourhood_slice[particle_index]:
            dip_x, dip_y, dip_z = part_pos
            m_dip_x, m_dip_y, m_dip_z = part_dip
            dipole_field = calculate_generic_dipole_zfield_at(dip_x, dip_y, dip_z, m_dip_x, m_dip_y, m_dip_z, particle_x, particle_y, particle_z)
            storage[0] = storage[0] + dipole_field[0]
            storage[1] = storage[1] + dipole_field[1]
            storage[2] = storage[2] + dipole_field[2]
    return storage


cdef (float,float,float) effective_dipole_field_at_point_full((int,float,float,float) particle, list slice):
    cdef (float,float,float) storage = (0,0,0)
    particle_index, particle_x, particle_y, particle_z = particle
    for part_id, part_pos, part_dip in slice:
        if part_id != particle_index:
            dip_x, dip_y, dip_z = part_pos
            m_dip_x, m_dip_y, m_dip_z = part_dip
            dipole_field = calculate_generic_dipole_zfield_at(dip_x, dip_y, dip_z, m_dip_x, m_dip_y, m_dip_z, particle_x, particle_y, particle_z)
            storage[0] = storage[0] + dipole_field[0]
            storage[1] = storage[1] + dipole_field[1]
            storage[2] = storage[2] + dipole_field[2]
    return storage


cdef (double,double,double) langevin_proj(float nula, float jedan, float dva, float dip_magnitude):
    cdef (double,double,double) dip_ret
    cdef float tri
    tri = sqrt(nula*nula+jedan*jedan+dva*dva)
    dipole_x = dip_magnitude*(1.0/tanh(dip_magnitude*tri)-1.0/(dip_magnitude*tri))*nula/tri
    dipole_y = dip_magnitude*(1.0/tanh(dip_magnitude*tri)-1.0/(dip_magnitude*tri))*jedan/tri
    dipole_z = dip_magnitude*(1.0/tanh(dip_magnitude*tri)-1.0/(dip_magnitude*tri))*dva/tri
    dip_ret = dipole_x, dipole_y, dipole_z
    return dip_ret


cpdef void magnetize(list dungeon_witch, tuple H, float dip_magnitude):
    cdef list slice_persists = [(x.id, x.pos, x.dip) for x in dungeon_witch]
    cdef list dipole_system_use_to_reset_values = [effective_dipole_field_at_point_full(
        (part.id, *part.pos), slice_persists) for part in dungeon_witch]
    cdef ParticleHandle part
    cdef (float,float,float) one
    for part,one in zip(dungeon_witch, dipole_system_use_to_reset_values):
        nula, jedan, dva = H[0]+one[0], H[1]+one[1], H[2]+one[2]
        part.dip = langevin_proj(nula, jedan, dva, dip_magnitude)
