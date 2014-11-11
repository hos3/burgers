'''Check where a PDE solver is stable by looking for proliferations of extrema.'''

import numpy as np
import scipy as sp
import pylab
import itertools
from burgersesLight import solve

roll = np.roll

def findextrema(u,strict=0):
	if strict:
		return np.where(np.sign(roll(u,-1)-u)*np.sign(u-roll(u,1)) < 0)[0]
	else:
		return np.where(np.sign(roll(u,-1)-u)*np.sign(u-roll(u,1)) <= 0)[0]


def istable(nt=10000,nxs=[200],dts=[.0004],etas=[.001],epss=[.005],offsets=[0.],flux=2,scheme=6) :
	'''Checks for stability of scheme "scheme" over the ranges of parameters 
		given by nxs, etas, and epss at time given by nt.'''
	
	lens = (len(nxs),len(dts),len(etas),len(epss),len(offsets))
	p = np.argsort(lens)			# This is the permutation of lens to make it ordered from low to high.
	
	nextrem = np.empty((lens[p[0]],lens[p[1]],lens[p[2]],lens[p[3]],lens[p[4]]))		# Ordered from low to high length for convenience
	
	for i in itertools.product(range(lens[0]),range(lens[1]),range(lens[2]),range(lens[3]),range(lens[4])):
		f = lambda t: np.sin(t) + offsets[i[4]]
		u = solve(nt=nt,nx=nxs[i[0]],dt=dts[i[1]],eta=etas[i[2]],eps=epss[i[3]],u_init=f,flux=flux,scheme=scheme,plots=0)
		nextrem[i[p[0]],i[p[1]],i[p[2]],i[p[3]],i[p[4]]] = len(findextrema(u))

	for (i,j,k) in itertools.product(range(lens[p[0]]),range(lens[p[1]]),range(lens[p[2]])):
		#pylab.figure()
		#pylab.plot(nextrem[i,j,k,:,:]>2)
		pylab.figure()
		pylab.imshow(nextrem[i,j,k,:,:])
	pylab.show()
	
	return nextrem
