# burgergence.py

from datetime import datetime as dt
from fourierBurgersesLight import solve
import pylab as pl
import numpy as np

def cvrg(d,dic=False,plots=True):
	'''Takes an iterable of dictionaries, and for each one the entries are
	expanded into an argument list for fourierBurgersesLight, which is then
	called and the results stored to be returned as the output of this funtion.
	If dic is set true, then the result is returned as a dictionary 
	(with keys being the entries of the input, d).  Otherwise the 
	result is a list of length len(d).'''
	
	s = []						# This will hold solution data
	
	for k in range(len(d)):
		T0 = dt.now()			# Time how long stuff takes
		a,*tmp = solve(plots=False,**d[k])
		s.append(a)
		T1 = dt.now()
		print( "Time elapsed for parameters "+str(d[k])+": "\
				+str((T1-T0).seconds + (T1-T0).microseconds/1e6) )
	
	if dic: s = dict(zip(d,s))
	if plots:
		for k in range(len(d)):
			if 'X0' in d[k]:
				X0 = d[k]['X0']
			else:
				X0 = 1
			x = np.linspace(0,X0,len(s[k]))
			pl.plot(x,s[k])
		pl.show()
	
	return s