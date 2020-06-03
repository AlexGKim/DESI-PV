#!/usr/bin/env python
import numpy.random
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt

class Fisher(object):

    def __init__(self,fiducial,sigmas,sigma_theta):
        self.set_fiducial(fiducial)
        self.invsigma2 = 1/sigmas**2
        self.invsigmatheta2 = 1/sigma_theta**2
        self.calcfids()

    def calcfids(self):
        self.tanh = numpy.tanh((self.theta_i-self.theta_0)/self.a)
        self.sech2 = 1-self.tanh**2

    def set_fiducial(self,fiducial):
        self.fiducial=fiducial
        self.v_inf=self.fiducial['v_inf']
        self.theta_i=self.fiducial['theta_i']
        self.theta_0=self.fiducial['theta_0']
        self.a = self.fiducial['a']
        self.calcfids()

    def fmatrix(self):
        N = len(sigmas)
        F = numpy.zeros((3+N,3+N))

        F[0,0] = (self.invsigma2*self.tanh**2).sum()
        for i in range(N):
            F[0,1+i]=self.v_inf/self.a * self.invsigma2[i]*self.tanh[i]*self.sech2[i]
        F[0,N+1] = - self.v_inf/self.a*(self.invsigma2*self.tanh*self.sech2).sum()
        F[0,N+2] = - self.v_inf/self.a**2*((self.theta_i-self.theta_0)*self.invsigma2*self.tanh*self.sech2).sum()

        for i in range(N):
          F[1+i,1+i]=self.v_inf**2/self.a**2 * self.invsigma2[i]*self.sech2[i]**2 + self.invsigmatheta2    
          F[1+i,N+1]= - self.v_inf**2/self.a**2* self.invsigma2[i] *self.sech2[i]**2
          F[1+i,N+2]= - self.v_inf**2/self.a**3 *(self.theta_i[i]-self.theta_0)* self.invsigma2[i] *self.sech2[i]**2

        F[N+1,N+1] = self.v_inf**2/self.a**2 * (self.invsigma2*self.sech2**2).sum()
        F[N+1,N+2] = self.v_inf**2/self.a**3*((self.theta_i-self.theta_0)*self.invsigma2*self.sech2**2).sum()

        F[N+2,N+2]= self.v_inf**2/self.a**4*((self.theta_i-self.theta_0)**2*self.invsigma2*self.sech2**2).sum()

        for i in range(0,3+N):
          for j in range(i+1,3+N):
            F[j,i]=F[i,j]
        return F

fiducial=dict()
fiducial['v_inf']=1.
fiducial['theta_i']=numpy.array([-1,-.5,0,.5,1])
fiducial['theta_0']=0.
fiducial['a']=0.75

sigmas=numpy.full(len(fiducial['theta_i']),0.02)
sigmas_theta = 0.02

f  = Fisher(fiducial, sigmas, sigmas_theta)
fm =f.fmatrix()
fm_inv = numpy.linalg.inv(fm)
# print(fm)
# print(fm[1,:])
# print(fm[3,:])
# wef
print(fm_inv)
print(numpy.diag(fm))
print(numpy.diag(fm_inv))