import numpy as np
cimport numpy as np
from libc.math cimport sqrt, cos, sin, pi, abs, acos, exp
from scipy.optimize import minimize
import random
cimport cython
from numpy cimport ndarray
from .particle_data cimport ParticleHandle



cdef class SW(object):

    cdef double Hk  # Hk = 2*K1/Js
    cdef double phi0
    cdef list nodes
    cdef double p0
    cdef double eps_phi
    cdef double kT_KVm
    cdef double p12
    cdef const double[:] e_k  

    def __init__(self, double Hk, double kT_KVm, double eps_phi=1e-3):  # , axis, **kwargs):
        self.Hk = Hk  # Hk = 2*K1/Js
        #self.e_k = np.array(axis) / norm(axis)
        self.phi0 = 0.
        self.nodes = []
        self.p0 = 1.
        self.eps_phi = eps_phi
        self.kT_KVm = kT_KVm
        self.p12 = -1
    
    
    cdef double phi(self, double theta, double h):
        
        cdef double max1
        cdef double max2
        fff = lambda phi: np.full(1, 0.25 - 0.25*cos(2*(phi-theta)) - h*cos(phi), dtype=float)
        dfff= lambda phi: np.full(1, 0.5*sin(2*(phi-theta))+h*sin(phi), dtype=float)
        invfff= lambda phi: np.full(1, -0.25 + 0.25*cos(2*(phi-theta)) + h*cos(phi), dtype=float)
        invdfff = lambda phi: np.full(1, -0.5*sin(2*(phi-theta))-h*sin(phi), dtype=float)
        cdef double min1 = minimize(fff, x0=self.phi0+self.eps_phi, jac=dfff, method="BFGS", tol=1e-15, options={'gtol': 1e-15})['x'][0]
        cdef double min2 = minimize(fff, x0=self.phi0+self.eps_phi - pi, jac=dfff, method="BFGS", tol=1e-15, options={'gtol': 1e-15})['x'][0]
        
        if min2 < -pi:
            min2 += 2. * pi
        elif min2 > pi:
            min2 -= 2. * pi

        if abs(min1 - min2) > 1.e-7:
            max1 = minimize(invfff, x0=self.phi0+self.eps_phi, jac=invdfff, method="BFGS", tol=1e-15, options={'gtol': 1e-15})['x'][0]
            max2 = minimize(invfff, x0=max1 - pi, jac=invdfff, method="BFGS", tol=1e-15, options={'gtol': 1e-15})['x'][0]
            if max1 < -pi:
                max1 += 2. * pi
            elif max1 > pi:
                max1 -= 2. * pi
            if max2 < -pi:
                max2 += 2. * pi
            elif max2 > pi:
                max2 -= 2. * pi
            b1 = fff(max1) - fff(min1)
            b2 = fff(max2) - fff(min1)
            p12 = 0.5 * self.p0 * (exp(-b1 / self.kT_KVm) + exp(-b2 / self.kT_KVm))
            self.p12 = p12
            if random.random() < p12:
                print("swap!")
                self.nodes = [min2, min1, max2, max1]
                sol = min2
            else:
                self.nodes = [min1, min2, max1, max2]
                sol = min1
        else:
            self.nodes = [min1]
            sol = min1

        self.phi0 = sol
        return sol

    cpdef void momtau(self, const double[:] H, list list_of_part_hndl):  # torque in reduced units (*mu0*ms*V for SI-units)
        
        cdef ParticleHandle part
        cdef const double[:] axis
        
        for part in list_of_part_hndl:
            axis = part.director
            self.e_k = axis
            H = np.array(H)
            normH = np.linalg.norm(H)
            e_h = H
            h = 0.

            if normH > 0.:
                e_h /= normH
                h = normH / self.Hk
            theta = acos(e_h.dot(self.e_k))
            axis = np.cross(e_h, self.e_k)

            if theta > pi / 2.:
                theta = pi - theta
                h = -h
                e_h = -e_h

            phi = self.phi(theta, h) % (2*pi)
            e_p = np.cross(np.cross(e_h, self.e_k), e_h)
            mom = e_h * cos(phi) + e_p * sin(phi)
            part.dip=1.732*mom
