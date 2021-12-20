import numpy as np
cimport numpy as np
from libc.math cimport sqrt, cos, sin, pi, abs, acos, exp
from scipy.optimize import minimize
import random
cimport cython
from numpy cimport ndarray
from .particle_data cimport ParticleHandle


cdef class SW(object):

    cdef float Hkinv  # Hk = 2*K1/Js
    cdef double[:] phi0
    cdef tuple[:] list_of_part_hndl
    cdef list nodes
    cdef float p0
    cdef float eps_phi
    cdef float kT_KVm_inv
    cdef float p12
    cdef float sat_magn
    cdef const double[:] e_k  

    def __init__(self, float Hk, float kT_KVm, float sat_magn, tuple[:] list_of_part_hndl, float eps_phi=1e-3):  # , axis, **kwargs):
        self.list_of_part_hndl = list_of_part_hndl
        self.Hkinv = 1/Hk  # Hk = 2*K1/Js
        self.phi0 = np.zeros(list_of_part_hndl.shape[0], dtype=float)
        self.nodes = []
        self.p0 = 1.
        self.eps_phi = eps_phi
        self.kT_KVm_inv = 1/kT_KVm
        self.p12 = -1
        self.sat_magn = sat_magn
    
    
    cdef float phi(self, float theta, float h, int index):
        
        cdef float max1
        cdef float max2
        fff = lambda phi: np.full(1, 0.25 - 0.25*cos(2*(phi-theta)) - h*cos(phi), dtype=float)
        dfff= lambda phi: np.full(1, 0.5*sin(2*(phi-theta))+h*sin(phi), dtype=float)
        invfff= lambda phi: np.full(1, -0.25 + 0.25*cos(2*(phi-theta)) + h*cos(phi), dtype=float)
        invdfff = lambda phi: np.full(1, -0.5*sin(2*(phi-theta))-h*sin(phi), dtype=float)
        cdef float min1 = minimize(fff, x0=self.phi0[index]+self.eps_phi, jac=dfff, method="BFGS", tol=1e-15, options={'gtol': 1e-15})['x'][0]
        cdef float min2 = minimize(fff, x0=self.phi0[index]+self.eps_phi - np.pi, jac=dfff, method="BFGS", tol=1e-15, options={'gtol': 1e-15})['x'][0]
        
        if min2 < -np.pi:
            min2 += 2. * np.pi
        elif min2 > np.pi:
            min2 -= 2. * np.pi

        if abs(min1 - min2) > 1.e-7:
            max1 = minimize(invfff, x0=self.phi0[index]+self.eps_phi, jac=invdfff, method="BFGS", tol=1e-15, options={'gtol': 1e-15})['x'][0]
            max2 = minimize(invfff, x0=max1 - pi, jac=invdfff, method="BFGS", tol=1e-15, options={'gtol': 1e-15})['x'][0]
            if max1 < -np.pi:
                max1 += 2. * np.pi
            elif max1 > np.pi:
                max1 -= 2. * np.pi
            if max2 < -np.pi:
                max2 += 2. * np.pi
            elif max2 > np.pi:
                max2 -= 2. * np.pi
            b1 = fff(max1) - fff(min1)
            b2 = fff(max2) - fff(min1)
            self.p12 = 0.5 * self.p0 * (exp(-b1 * self.kT_KVm_inv) + exp(-b2 * self.kT_KVm_inv))       
            if random.random() < self.p12:
                print("swap!")
                self.nodes = [min2, min1, max2, max1]
                sol = min2
            else:
                self.nodes = [min1, min2, max1, max2]
                sol = min1
        else:
            self.nodes = [min1]
            sol = min1

        self.phi0[index] = sol
        return sol

    cpdef void momtau(self, const double[:] H):  # torque in reduced units (*mu0*ms*V for SI-units)
        cdef ParticleHandle part_real, part_virt
        cdef const double[:] axis
        cdef int index
        cdef np.ndarray H_array = np.asarray(H)
        cdef float normH = np.linalg.norm(H_array)

        e_h = H_array/normH
        h = normH * self.Hkinv
        
        for index in range(self.list_of_part_hndl.shape[0]):
            part_real, part_virt = self.list_of_part_hndl[index]
            axis = part_real.director
            self.e_k = axis

            theta = acos(e_h.dot(self.e_k))
            axis = np.cross(e_h, self.e_k)

            if theta > np.pi / 2.:
                theta = np.pi - theta
                h = -h
                e_h = -e_h

            phi = self.phi(theta, h, index) % (2*np.pi)
            rotaxis = np.cross(e_h, self.e_k)
            rotaxis /= np.linalg.norm(rotaxis)
            mom = e_h * np.cos(phi) + np.cross(rotaxis, e_h) * np.sin(phi)
            part_virt.dip=self.sat_magn*mom
