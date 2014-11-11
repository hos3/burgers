'''Find zero crossings'''

import numpy as np

def zeros(a):
	'''Finds zero crossings of numpy vector a.
	Zeros are considered to be both true zeros or places where the sign flips from + to - or - to +.'''
	s = np.sign(a)
	s = np.concatenate((s[[0]]==0,np.logical_or(s[1:]==0,s[1:]==-s[:-1])))
	return np.where(s)