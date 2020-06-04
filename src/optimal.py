#!/usr/bin/env python
import numpy.random
from scipy.optimize import minimize
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

  def set_sigmas(self,sigmas):
    self.invsigma2 = 1/sigmas**2    

  def fmatrix(self):
    # print(self.theta_i)

    N = len(self.invsigma2)
    F = numpy.zeros((4+N,4+N))

    F[0,0] = (self.invsigma2*self.tanh**2).sum()
    for i in range(N):
        F[0,1+i]=self.v_inf/self.a * self.invsigma2[i]*self.tanh[i]*self.sech2[i]
    F[0,N+1] = - self.v_inf/self.a*(self.invsigma2*self.tanh*self.sech2).sum()
    F[0,N+2] = - self.v_inf/self.a**2*((self.theta_i-self.theta_0)*self.invsigma2*self.tanh*self.sech2).sum()
    F[0,N+3] = (self.invsigma2*self.tanh).sum()

    for i in range(N):
      F[1+i,1+i]=self.v_inf**2/self.a**2 * self.invsigma2[i]*self.sech2[i]**2 + self.invsigmatheta2    
      F[1+i,N+1]= - self.v_inf**2/self.a**2* self.invsigma2[i] *self.sech2[i]**2
      F[1+i,N+2]= - self.v_inf**2/self.a**3 *(self.theta_i[i]-self.theta_0)* self.invsigma2[i] *self.sech2[i]**2
      F[1+i,N+3]= self.v_inf/self.a * self.invsigma2[i]*self.sech2[i]

    F[N+1,N+1] = self.v_inf**2/self.a**2 * (self.invsigma2*self.sech2**2).sum()
    F[N+1,N+2] = self.v_inf**2/self.a**3*((self.theta_i-self.theta_0)*self.invsigma2*self.sech2**2).sum()
    F[N+1,N+3] = - self.v_inf/self.a * (self.invsigma2*self.sech2).sum()

    F[N+2,N+2]= self.v_inf**2/self.a**4*((self.theta_i-self.theta_0)**2*self.invsigma2*self.sech2**2).sum()
    F[N+2,N+3]= -self.v_inf**2/self.a**2*((self.theta_i-self.theta_0)*self.invsigma2*self.sech2).sum()

    F[N+3,N+3] = self.invsigma2.sum()

    for i in range(0,4+N):
      for j in range(i+1,4+N):
        F[j,i]=F[i,j]
    return F


def sersicProfile(fiducial, n=1.):
  b=2*n-1./3
  return numpy.exp(-b*((numpy.abs(fiducial['theta_i']-fiducial['theta_0'])/fiducial['a'])**(1./n) -1))

def sigmas_v(norm, background, fiducial, profile, **kwargs):
  gflux = profile(fiducial, **kwargs)
  return norm* numpy.sqrt(background + gflux)/gflux

def x_to_theta(x):
  x[1:] = numpy.exp(x[1:]/100)
  theta_i = numpy.array(x)
  for i in range(1,len(x)):
    theta_i[i]=theta_i[i-1]+x[i]
  return theta_i

def theta_to_x(theta):
  x=numpy.array(theta)
  delta = numpy.roll(theta,-1)-theta
  x[1:]= 100*numpy.log(delta[:-1])
  return x

def objective(x, fisher, profile, index=0, zero=False):

  if zero:
    x=numpy.append(x,0)

  theta_i = x_to_theta(x)

  fiducial = fisher.fiducial
  fiducial['theta_i'] = theta_i

  fisher.set_fiducial(fiducial)

  fisher.set_sigmas(sigmas_v(0.1, 0, fiducial, profile))
  f = fisher.fmatrix()
  ans = numpy.sqrt(numpy.linalg.inv(f)[index,index])
  return ans

def main():

  # values of the fiducial parameters
  fiducial=dict()
  fiducial['v_inf']=1.
  fiducial['theta_i']=numpy.array([-1,-0.5,.5,1])
  fiducial['theta_0']=0.
  fiducial['a']=1.

  #values of the measurement errors
  sigmas = sigmas_v(0.1, 0, fiducial, sersicProfile)
  sigmas_theta = 0.02

  f  = Fisher(fiducial, sigmas, sigmas_theta)


  x0=theta_to_x(numpy.array([-1,-0.5,.5,1]))
  res = minimize(objective, x0, method='powell',args=(f, sersicProfile), options={'xtol': 1e-8, 'disp': False})
  print(x_to_theta(res.x), res.fun)

  x0=theta_to_x(numpy.array([-1,-0.5,.5,1]))
  res = minimize(objective, x0, method='powell',args=(f, sersicProfile,-1), options={'xtol': 1e-8, 'disp': False})
  print(x_to_theta(res.x), res.fun)

  # x0=theta_to_x(numpy.array([-1,-0.67,-0.33,0]))
  # res = minimize(objective, x0, method='powell',args=(f,-1), options={'xtol': 1e-8, 'disp': False})
  # print(x_to_theta(res.x), res.fun)
  # wefwe

  x0=theta_to_x(numpy.array([-1,-0.5,.5,1]))
  res = minimize(objective, x0, method='powell',args=(f, sersicProfile,0,True), options={'xtol': 1e-8, 'disp': False})
  print(x_to_theta(res.x), res.fun)

  x0=theta_to_x(numpy.array([-1,-0.5,.5,1]))
  res = minimize(objective, x0, method='powell',args=(f, sersicProfile,-1,True), options={'xtol': 1e-8, 'disp': False})
  print(x_to_theta(res.x), res.fun)

if __name__ == "__main__":
    # execute only if run as a script
    main()