'''Front tracking for Burgers equations'''

import numpy as np
import scipy as sp
import pylab
from burgerses import solve

def findz(a):
	'''Finds zero crossings of numpy vector a.
	Zeros are considered to be both true zeros or places where the sign flips from + to - or - to +.'''
	s = np.sign(a)
	s = np.logical_or(s==0,s==-np.roll(s,1))		# If it's a true 0 or if the sign is opposite of the previous step
	return np.where(s)[0]

def trackz(flux,eta,nt=20000,tstride=250,scheme=1):
	'''Tracks zeros in Burgers' equation with flux given by 'flux' for time nt.
		eta is a vector of etas for which to track the fronts.
		tstride specifies the interval for tracking zeros (under the assumption
		that it is not necessary to find the zeros at every time step).'''
	
	ranj = range(0,nt,tstride)			# These are the timesteps at which to evaluate the zeros
	
	for e in range(len(eta)):			# For each value of eta
		print('Processing for eta == ',end=''); print(eta[e])
		
		u = solve(nt=nt,eta=eta[e],flux=flux,scheme=scheme,plots=0)		# Get the solution
		zros = []						# This will store all the zero locations at all (relevant) times for this value of eta
		idx = []						# This will contain the times corresponding to each zros entry
		for k in ranj:
			nzros = findz(u[:,k])		# Zero locations at this time step
			zros.append(nzros)			# Update zeros for this value of eta
			idx += [k] * len(nzros)		# Store time step for each zero
		zros = np.concatenate(zros)		# Combine all zeros into one list
		idx = np.array(idx)				# Combine all time step info too
		pylab.plot(idx,zros,'o')
	
	pylab.show()