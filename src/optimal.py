#!/usr/bin/env python
import numpy.random
from scipy.optimize import minimize
import scipy.stats
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.titlesize': 'large'})
plt.rcParams.update({'axes.labelsize': 'large'})

class PositionFisher(object):

  def __init__(self,fiducial,noiseModel): # sigmas,sigma_theta):
    self.fiducial=fiducial
    self.noiseModel = noiseModel

  def fmatrix(self):
    v_inf=self.fiducial['v_inf']
    theta_i=self.fiducial['theta_i']
    theta_0=self.fiducial['theta_0']
    a = self.fiducial['a']
    # the nominal galaxy velocity value doesn't matter

    tanh = numpy.tanh((theta_i-theta_0)/a)
    sech2 = 1-tanh**2

    invsigma2 = 1/self.noiseModel.sigmas_v(theta_i-theta_0)**2
    invsigmatheta2 = 1/self.noiseModel.sigmas_theta**2

    N = len(invsigma2)
    F = numpy.zeros((4+N,4+N))

    F[0,0] = (invsigma2*tanh**2).sum()
    for i in range(N):
        F[0,1+i]=v_inf/a * invsigma2[i]*tanh[i]*sech2[i]
    F[0,N+1] = - v_inf/a*(invsigma2*tanh*sech2).sum()
    F[0,N+2] = - v_inf/a**2*((theta_i-theta_0)*invsigma2*tanh*sech2).sum()
    F[0,N+3] = (invsigma2*tanh).sum()

    for i in range(N):
      F[1+i,1+i]=v_inf**2/a**2 * invsigma2[i]*sech2[i]**2 + invsigmatheta2    
      F[1+i,N+1]= - v_inf**2/a**2* invsigma2[i] *sech2[i]**2
      F[1+i,N+2]= - v_inf**2/a**3 *(theta_i[i]-theta_0)* invsigma2[i] *sech2[i]**2
      F[1+i,N+3]= v_inf/a * invsigma2[i]*sech2[i]

    F[N+1,N+1] = v_inf**2/a**2 * (invsigma2*sech2**2).sum()
    F[N+1,N+2] = v_inf**2/a**3*((theta_i-theta_0)*invsigma2*sech2**2).sum()
    F[N+1,N+3] = - v_inf/a * (invsigma2*sech2).sum()

    F[N+2,N+2]= v_inf**2/a**4*((theta_i-theta_0)**2*invsigma2*sech2**2).sum()
    F[N+2,N+3]= -v_inf**2/a**2*((theta_i-theta_0)*invsigma2*sech2).sum()

    F[N+3,N+3] = invsigma2.sum()

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
#
# The noise in the pointing of the fibers is
# sigmas_theta: fiber pointing error

class MaxNoiseModel(object):

  def __init__(self,theta_max, sigma_v_max, background, profile, sigmas_theta, **kwargs):
    self.theta_max = theta_max
    self.sigma_v_max = sigma_v_max
    self.background = background
    self.profile = profile
    self.sigmas_theta=sigmas_theta
    if kwargs:
      self.kwargs = kwargs
    else:
      self.kwargs = dict()

  def sigmas_v(self, theta):
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

# index 0 is 'v_inf'
# if one of the data is zero then force it in there
def objective(x, fisher, index=0, addZero=False):

  if addZero:
    x=numpy.append(x,0)

  theta_i = x_to_theta(x)

  fisher.fiducial['theta_i'] = theta_i

  f = fisher.fmatrix()
  # ans = numpy.linalg.inv(f)[index,index]
  # return ans
  ans = numpy.sqrt(numpy.linalg.inv(f)[index,index])
  return ans

def defaultFisher():

  # values of the fiducial parameters
  fiducial=dict()
  fiducial['v_inf']=1.
  fiducial['theta_i']=numpy.array([-1,-0.5,.5,1])
  fiducial['theta_0']=0.
  fiducial['a']=1.
  fiducial['v_0']=0.

  #values of the measurement errors
  # sigmas = sigmas_v(0.1, 0, fiducial, sersicProfile)
  sigma_v_max=1.    # units don't matter for optimization
  theta_max=4       # all theta units in optical R_e
  background=0.
  sigmas_theta = 0.01 # say an uncertainty of 0.2" for an R_e=30"
  noiseModel = MaxNoiseModel(theta_max, sigma_v_max, background, sersicProfile, sigmas_theta)

  return PositionFisher(fiducial, noiseModel)


def vary_a():
  f = defaultFisher()
  theta_max = f.noiseModel.theta_max
  theta_start  = numpy.array([-0.9*theta_max,0.45*theta_max,0.9*theta_max])
  x0=theta_to_x(theta_start)

  a_s =  numpy.arange(1,4.01,0.1)
  ans=[]
  for a_ in a_s:
    f.fiducial['a']=a_
    res = minimize(objective, x0, method='Nelder-Mead',args=(f, 0,True), options={'xtol': 1e-16, 'ftol':1e-16, 'disp': False})
    ans.append(x_to_theta(res.x))

  ans = numpy.array(ans)

  plt.plot(ans,a_s)
  plt.axvline(0)
  plt.xlabel(r'$\theta [R_e]$')
  plt.ylabel(r'$a [R_e]$')
  plt.ylim((a_s[0],a_s[-1]))
  plt.title(r'$\theta^\mathrm{max} = 4 [R_e]$, $\sigma_\theta=0.01 [R_e]$')
  plt.savefig('vary_a.pdf')
  plt.clf()
  # print(x_to_theta(res.x), res.fun)


  # x0=theta_to_x(theta_start)
  # res = minimize(objective, x0, method='powell',args=(f, noiseModel), options={'xtol': 1e-8, 'disp': False})
  # print(x_to_theta(res.x), res.fun)

  # x0=theta_to_x(theta_start)
  # res = minimize(objective, x0, method='powell',args=(f, noiseModel,-1), options={'xtol': 1e-8, 'disp': False})
  # print(x_to_theta(res.x), res.fun)



  # x0=theta_to_x(theta_start)
  # res = minimize(objective, x0, method='powell',args=(f, noiseModel,-1,True), options={'xtol': 1e-8, 'disp': False})
  # print(x_to_theta(res.x), res.fun)

def vary_theta_max():
  f = defaultFisher()
  f.fiducial['a']=4.


  theta_maxs =  numpy.arange(1,4.01,0.1) 
  ans=[]
  for theta_max in theta_maxs:
    f.noiseModel.theta_max=theta_max
    theta_start  = numpy.array([-0.9*theta_max,0.45*theta_max,0.9*theta_max])
    x0=theta_to_x(theta_start)
    res = minimize(objective, x0, method='Nelder-Mead',args=(f, 0,True), options={'xtol': 1e-16, 'ftol':1e-16, 'disp': False})
    ans.append(x_to_theta(res.x))

  ans = numpy.array(ans)

  plt.plot(ans,theta_maxs)
  plt.axvline(0)
  plt.xlabel(r'$\theta [R_e]$')
  plt.ylabel(r'$\theta^\mathrm{max} [R_e]$')
  plt.ylim((theta_maxs[0],theta_maxs[-1]))
  plt.title(r'$a = 4 [R_e]$, $\sigma_\theta=0.01$  [R_e]')
  plt.savefig('vary_theta_max.pdf')
  plt.clf()

def vary_sigma_theta():
  f = defaultFisher()
  theta_max = f.noiseModel.theta_max
  theta_start  = numpy.array([-0.9*theta_max,0.45*theta_max,0.9*theta_max])
  x0=theta_to_x(theta_start)

  sigmas_thetas = 10**numpy.arange(-3,-0.99,0.1)
  ans=[]
  for sigmas_theta in sigmas_thetas:
    f.noiseModel.sigmas_theta=sigmas_theta
    res = minimize(objective, x0, method='Nelder-Mead',args=(f, 0,True), options={'xtol': 1e-16, 'ftol':1e-16, 'disp': False})
    ans.append(x_to_theta(res.x))

  ans = numpy.array(ans)

  plt.plot(ans,sigmas_thetas)
  plt.axvline(0)
  plt.xlabel(r'$\theta [R_e]$')
  plt.ylabel(r'$\sigma_\theta [R_e]$')
  plt.ylim((0,sigmas_thetas[-1]))
  plt.title(r'$\theta^\mathrm{max} = 4 [R_e]$, $a = 4 [R_e]$')
  plt.savefig('vary_sigma_theta.pdf')
  plt.clf()

def main():
  vary_theta_max()
  vary_a()
  vary_sigma_theta()

if __name__ == "__main__":
    # execute only if run as a script
    main()