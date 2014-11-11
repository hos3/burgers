'''burgerses.py
Solves modified Burgers' equations by a variety of schemes. 
'''

import scipy as np						# Started out using numpy, then switched to scipy for consistency with expdelta
import scipy as sp
import pylab
from scipy import interpolate as interpolate

roll = np.roll							# Used often

# Default values:
eps0 = .005								# epsilon
eta0 = .005								# eta (viscosity)
nx0 = 200								# number of x grid points
nt0 = 20000								# number of time grid points
dx0 = 2.*np.pi/nx0						# x grid spacing
dt0 = .0004								# time grid spacing
u_init0 = lambda t: np.sin(t) + .1		# Initial distribution

def Dx(f,dx):							# Symmetric derivative
	return (roll(f,-1) - roll(f,1))/(2.*dx)

def Dx1(f,dx,pm):						# Asymmetric derivative: pm==-1 differences upwards, pm==1 differences downwards
	return (f-roll(f,pm))/(pm*dx)

def solve(nx=nx0, nt=nt0, dx=None, dt=dt0, u_init=u_init0, eps=eps0, eta=eta0, flux=0, scheme=1, plots=1):
	'''Solves du/dt = -dJ/dx + eta * uxx.
		Returns u, a shape (nx,) array with u[j] = the solution at x=j*dx & t=nt*dt.
		So this is a "light" version of burgerses.solve, which returns only the final state rather than
		the full time evolution of u.  So this is more suitable for very long times, for example. 
		J is set via the flux parameter; flux==0 gives J==.5*u**2, flux==1 gives J==.5*u**2*exp(-eps*ux**2),
			and flux==2 gives J==.5*u**2/(1.+eps*ux**2); if flux is a function, J = flux
		A variety of differencing schemes are available, via the scheme parameter.	
		The plots keyword determines whether to plot nothing (plots==0) or just the final state (plots==1).'''
	
	if dx is None: dx = 2.*np.pi/nx		# For convenience, since [0,1] is the main inteval I've been using
	
	x = np.arange(nx)*dx
	u = u_init(x)					# Initial conditions
	
	# Determine flux function:
	if flux==0:						# Unmodified Burgers' equation
		getJ = lambda v, vx : .5 * v**2
	elif flux==1:					# Burgers' equation with exp
		getJ = lambda v, vx : .5 * v**2 * np.exp(-eps*vx**2)
	elif flux==2:					# Burgers' equation with polynomial
		getJ = lambda v, vx : .5 * v**2 / (1.+eps*vx**2)
	elif callable(flux):			# Custom flux
		getJ = flux
	
	if scheme==1:		# This is the scheme used in expdelta
		for k in range(nt-1):
			um = roll(u,1); up = roll(u,-1)	# Here and elsewhere, p stands for "plus 1" and m ...
			ux = Dx(u,dx)								#...for "minus 1", in the context of index increments
			J = getJ(u,ux)
			u+= - dt*Dx(J,dx) + dt*eta*(up+um-u*2.)/(dx**2)		# Artificial viscosity
	elif scheme==2:		# Absolute downhill differencing
		for k in range(nt-1):
			um = roll(u,1); up = roll(u,-1); up2 = roll(up,-1);
			uxm = (u-um)/dx; uxp = (up2-up)/dx;		# Left and right derivatives
			msk = abs(up) > abs(u)						# Determine which way to difference
			J = np.empty(up.shape)							# Flux
			J[msk] = getJ(u[msk],uxm[msk]); J[~msk] = getJ(up[~msk],uxp[~msk]);		# Get flux from right or left according to msk
			u += - dt*(Dx1(J,dx,1)) + dt*eta*(up+um-u*2.)/(dx**2)		# Update u
	
	# For the next bunch, it turns out that if you conditionally difference u and NOT ux, the results are
	# much better than if you conditionally difference both of them.  I've included both options for 
	# comparison.  Schemes 3 and 5 conditionally difference both u and ux, while  4 and 6 do only u. 
	
	# The idea of each of the following is identical to scheme 2, so I'll forgo excessive commenting. 
	elif scheme==3:		# Small gradient-norm differencing 1: condition u and ux
		for k in range(nt-1):
			um = roll(u,1); up = roll(u,-1); up2 = roll(up,-1);
			uxm = (u-um)/dx; uxp = (up2-up)/dx;
			msk = abs(up2-up) > abs(u-um)
			J = np.empty(up.shape)
			J[msk] = getJ(u[msk],uxm[msk]); J[~msk] = getJ(up[~msk],uxp[~msk]);
			u += - dt*(Dx1(J,dx,1)) + dt*eta*(up+um-u*2.)/(dx**2)
	elif scheme==4:		# Small gradient-norm differencing 2: condition u only
		for k in range(nt-1):
			um = roll(u,1); up = roll(u,-1); up2 = roll(up,-1);
			uxm = (u-um)/dx; uxp = (up2-up)/dx; ux = (up-u)/dx;
			msk = abs(up2-up) > abs(u-um)
			J = np.empty(up.shape)
			J[msk] = getJ(u[msk],ux[msk]); J[~msk] = getJ(up[~msk],ux[~msk]);
			u += - dt*(Dx1(J,dx,1)) + dt*eta*(up+um-u*2.)/(dx**2)
	elif scheme==5:		# Upwind differencing 1: condition u and ux
		for k in range(nt-1):
			um = roll(u,1); up = roll(u,-1); up2 = roll(up,-1);
			uxp = (up2-up)/dx; uxm = (u-um)/dx;
			J = getJ(up,uxp); Jm = getJ(u,uxm)
			msk = np.sign(up-um)*np.sign(-Dx1(J+Jm,dx,1)) == -1
			J[msk] = Jm[msk]
			u += - dt*(Dx1(J,dx,1)) + dt*eta*(up+um-u*2.)/(dx**2)
	elif scheme==6:		# Upwind differencing 2: condition u only
		for k in range(nt-1):
			um = roll(u,1); up = roll(u,-1); up2 = roll(up,-1);
			uxp = (up2-up)/dx; uxm = (u-um)/dx; ux = (up-u)/dx;
			J = getJ(up,ux); Jm = getJ(u,ux)
			msk = np.sign(up-um)*np.sign(-Dx1(J+Jm,dx,1)) == -1
			J[msk] = Jm[msk]
			u += - dt*(Dx1(J,dx,1)) + dt*eta*(up+um-u*2.)/(dx**2)
	if plots==1:
		pylab.plot(u);pylab.show();
	
	return u


def converj(lim, T, Xmax=2*np.pi, u_init=u_init0, flux=2, scheme=6, plots=1):
	# Looks at convergence of solutions to modified Burgers' eq. in limit specified by "lim".
	# "lim" should specify a sequence of tuples (dx, dt, eps, eta), and thus should
	# either be an iterable (with len attribute), or else a 2-tuple (f,s) where f is a function
	# outputting (dx, dt, eps, eta) and s is a sequence at which to evaluate f.
	# T is the time slice at which to compare the solutions. 
	# if plots: plot output u
	
	if callable(lim[0]):					# Convert lim to standard format
		lim = [lim[0](i) for i in lim[1]]
	
	dx_min = lim[-1][0]						# Assuming that dx decreases monotonically under the limit
	nx_max = int(np.ceil(Xmax/dx_min))		# Maximum number of spacial steps
	xterp = np.linspace(0,Xmax,nx_max+1)	# Grid for comparison
	u = np.empty( (len(lim),),dtype='object' )		# Array for storing interpolated solutions under the limit
	xs = u.copy()							# Array for storing x coordinates of solutions
	
	from keyboard import keyboard
	
	for i in range(len(lim)):
		(dx,dt,eps,eta) = lim[i]
		nx = int( np.ceil(Xmax/dx) )
		dx = Xmax/float(nx)			# float is for python 2.7 compatability
		nt = int( np.ceil(T/dt) )
		dt = T/float(nt)
		print('Evaluating at (dx,dt,eps,eta) = ({:.2E},{:.2E},{:.2E},{:.2E})'.format(dx,dt,eps,eta))
		
		u0 = solve( nx, nt, dx, dt, u_init, eps, eta, flux, scheme, 0)
		xs[i] = np.arange(nx)*dx
		u[i] = u0
		
	if plots:
		for i in range(len(xs)):
			pylab.plot(xs[i],u[i])
			pylab.show()
	
	return u, xs


def countpeaks(u,type='max'):
	'''Counts the number of extrema of u, assuming periodicity.
	'type' determines whether to look for maxima, minima, or both'''
	
	n = 0
	if type=='max' or type=='both':
		n+=np.sum( (u>roll(u,-1)) & (u>=roll(u,1)) )
	
	if type=='min' or type=='both':
		n+=np.sum( (u<roll(u,-1)) & (u<=roll(u,1)) )
	return n

from itertools import product

def stabil(dx,dt,eps,eta,T,flux=2,scheme=6,plots=1):
	'''Checks stability at a variety of values of (dx,dt,eps,eta).
	All input parameters are taken to be vectors.
	X range is 2*np.pi and time at which to check stability is T.'''
	
	# First make sure arrays are arrays:
	dx=np.array(dx).flatten(); dt=np.array(dt).flatten(); eps=np.array(eps).flatten(); eta=np.array(eta).flatten();
	
	stab = np.empty((len(dx),len(dt),len(eps),len(eta)),dtype='int')
	
	for (i,j,k,l) in product(range(len(dx)),range(len(dt)),range(len(eps)),range(len(eta))):
		nx = np.ceil(2*np.pi/dx[i]); nt = np.ceil(T/dt[j]);
		u = solve(int(nx), int(nt), None, T/nt, u_init0, eps[k], eta[l], flux, scheme, 0)
		stab[i,j,k,l] = countpeaks(u)
		
		print('For (dx,dt,eps,eta) = ({:.2E},{:.2E},{:.2E},{:.2E}), # of maxima = {:n}'.format(\
				2*np.pi/nx, T/nt, eps[k],eta[l], stab[i,j,k,l]) )
	
	if plots:
		for (k,l) in product(range(len(eps)),range(len(eta))):
			pylab.figure()
			pylab.imshow(np.log(stab[:,:,k,l]+1).T)
			pylab.colorbar()
	
	return stab


def stabil2(dx,dt,eps,eta,T,flux=2,scheme=6,plots=1):
	'''Checks stability at a variety of values of (dx,dt,eps,eta).
	All input parameters are taken to be vectors.
	X range is 2*np.pi and time at which to check stability is T.
	For best efficiency, sort dt in ascending order.'''
	
	# First make sure arrays are arrays:
	dx=np.array(dx).flatten(); dt=np.array(dt).flatten(); eps=np.array(eps).flatten(); eta=np.array(eta).flatten();
	
	stab = np.empty((len(dx),len(dt),len(eps),len(eta)),dtype='bool')
	
	for (i,j,k,l) in product(range(len(dx)),range(len(dt)),range(len(eps)),range(len(eta))):
		if (j>0) and (dt[j]>dt[j-1]) and (not stab[i,j-1,k,l]): 
			stab[i,j,k,l] = False
			continue
		if (j>0) and (dt[j]<dt[j-1]) and (stab[i,j-1,k,l]):
			stab[i,j,k,l] = True
			continue
		if (i>0) and (dx[i]<dx[i-1]) and (not stab[i-1,j,k,l]):
			stab[i,j,k,l] = False
			continue
		if (i>0) and (dx[i]>dx[i-1]) and (stab[i-1,j,k,l]):
			stab[i,j,k,l] = True
			continue
		
		nx = np.ceil(2*np.pi/dx[i]); nt = np.ceil(T/dt[j]);
		u = solve(int(nx), int(nt), None, T/nt, u_init0, eps[k], eta[l], flux, scheme, 0)
		stab[i,j,k,l] = 0<countpeaks(u)<=2
		
		print('For dx={:.2E}, dt={:.2E}, eps={:.2E}, eta={:.2E}: stable? {:n}\n'.format(\
				2*np.pi/nx, T/nt, eps[k],eta[l], stab[i,j,k,l]) )
	
	if plots:
		for (k,l) in product(range(len(eps)),range(len(eta))):
			pylab.figure()
			pylab.imshow((stab[:,:,k,l].T))
			pylab.colorbar()
	
	return stab
