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

# The noise based on signal to noise so depends on
#
# profile :     shape of the galaxy
# backgound:    background flux
# 
# The noise is normalized based on the
# sigma_v_max:  the maximum possible noise
# theta_max:    the maximum radius that has the maximum noise

class MaxNoiseModel(object):

  def __init__(self,theta_max, sigma_v_max, background, profile, **kwargs):
    self.theta_max = theta_max
    self.sigma_v_max = sigma_v_max
    self.background = background
    self.profile = profile
    if kwargs:
      self.kwargs = kwargs
    else:
      self.kwargs = dict()

  def sigmas(self, theta):
    minflux = self.profile(self.theta_max, **self.kwargs)
    ans = numpy.zeros(len(theta))
    w = numpy.abs(theta) < self.theta_max
    gflux = self.profile(theta[w], **self.kwargs)
    ans[w] = numpy.sqrt((gflux+self.background)/(minflux+self.background)) /gflux * minflux
    w = numpy.abs(theta) >= self.theta_max
    ans[w] = numpy.exp(5*(numpy.abs(theta[w]) - self.theta_max))
    ans = self.sigma_v_max*ans
    return ans

  @staticmethod
  def test():
    sigma_v_max=10.
    theta = numpy.arange(-4.1,4.101,0.01)
    theta_max=4
    background=1
    sig = MaxNoiseModel(theta_max, sigma_v_max, background, sersicProfile)
    sigs = sig.sigmas(theta)
    # plt.plot(theta, sig.sigmas(theta))
    sig.background=0
    sigs = sigs/sig.sigmas(theta)
    plt.plot(theta, sigs)
    plt.show()

def sersicProfile(theta_Re, n=1.):
  b=2*n-1./3
  return numpy.exp(-b*(numpy.abs(theta_Re)**(1./n) -1))

# the objective function variable 'x' are not theta, the transformation between the two are
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

def objective(x, fisher, noiseModel, index=0, addZero=False):

  if addZero:
    x=numpy.append(x,0)

  theta_i = x_to_theta(x)

  fiducial = fisher.fiducial
  fiducial['theta_i'] = theta_i

  fisher.set_fiducial(fiducial)

  fisher.set_sigmas(noiseModel.sigmas(fiducial['theta_i']-fiducial['theta_0']))
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
  # sigmas = sigmas_v(0.1, 0, fiducial, sersicProfile)
  sigma_v_max=10.
  theta_max=4
  background=1
  noiseModel = MaxNoiseModel(theta_max, sigma_v_max, background, sersicProfile)
  # sigmas = sigmas_v(fiducial['theta_i']-fiducial['theta_0'], 2, 0.1, 0, sersicProfile)
  sigmas = noiseModel.sigmas(fiducial['theta_i']-fiducial['theta_0'])
  sigmas_theta = 0.02

  f  = Fisher(fiducial, sigmas, sigmas_theta)

  x0=theta_to_x(numpy.array([-1,-0.5,.5,1]))
  res = minimize(objective, x0, method='powell',args=(f, noiseModel), options={'xtol': 1e-8, 'disp': False})
  print(x_to_theta(res.x), res.fun)

  x0=theta_to_x(numpy.array([-1,-0.5,.5,1]))
  res = minimize(objective, x0, method='powell',args=(f, noiseModel,-1), options={'xtol': 1e-8, 'disp': False})
  print(x_to_theta(res.x), res.fun)

  x0=theta_to_x(numpy.array([-1,-0.5,.5,1]))
  res = minimize(objective, x0, method='powell',args=(f, noiseModel,0,True), options={'xtol': 1e-8, 'disp': False})
  print(x_to_theta(res.x), res.fun)

  x0=theta_to_x(numpy.array([-1,-0.5,.5,1]))
  res = minimize(objective, x0, method='powell',args=(f, noiseModel,-1,True), options={'xtol': 1e-8, 'disp': False})
  print(x_to_theta(res.x), res.fun)

if __name__ == "__main__":
    # execute only if run as a script
    main()