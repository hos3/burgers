'''burgerses.py
Solves modified Burgers' equations by a variety of schemes. 
'''

import scipy as np						# Started out using numpy, then switched to scipy for consistency with expdelta
import scipy
import pylab

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
		Returns u, a shape (nx,nt) array with u[j,k] = the solution at x=j*dx & t=k*dt.
		J is set via the flux parameter; flux==0 gives J==.5*u**2, flux==1 gives J==.5*u**2*exp(-eps*ux**2),
			and flux==2 gives J==.5*u**2/(1.+eps*ux**2); if flux is a function, J = flux
		A variety of differencing schemes are available, via the scheme parameter.	
		The plots keyword determines whether to plot nothing (plots==0), the full time evolution (plots==1)
		or just the final state (plots==2).'''
	
	if dx is None: dx = 2.*np.pi/nx		# For convenience, since [0,1] is the main inteval I've been using
	
	u = np.empty((nx,nt))
	x = np.arange(nx)*dx
	u[:,0] = u_init(x)				# Initial conditions
	
	# Determine flux function:
	if flux==0:						# Unmodified Burgers' equation
		getJ = lambda u, ux : .5 * u**2
	elif flux==1:					# Burgers' equation with exp
		getJ = lambda u, ux : .5 * u**2 * np.exp(-eps*ux**2)
	elif flux==2:					# Burgers' equation with polynomial
		getJ = lambda u, ux : .5 * u**2 / (1.+eps*ux**2)
	elif callable(flux):			# Custom flux
		getJ = flux
	
	if scheme==1:		# This is the scheme used in expdelta
		for k in range(nt-1):
			um = roll(u[:,k],1); up = roll(u[:,k],-1)	# Here and elsewhere, p stands for "plus 1" and m ...
			um2 = roll(um,1); up2 = roll(up,-1)
			ux = Dx(u[:,k],dx)								#...for "minus 1", in the context of index increments
			J = getJ(u[:,k],ux)
			u[:,k+1] = u[:,k] - dt*Dx(J,dx) - dt*eta*(up2-4.*up+6.*u[:,k]-4.*um+um2)/(dx**4)		# Artificial viscosity
	elif scheme==2:		# Absolute downhill differencing
		for k in range(nt-1):
			um = roll(u[:,k],1); up = roll(u[:,k],-1); 
			um2 = roll(um,1); up2 = roll(up,-1)
			uxm = (u[:,k]-um)/dx; uxp = (up2-up)/dx;		# Left and right derivatives
			msk = abs(up) > abs(u[:,k])						# Determine which way to difference
			J = np.empty(up.shape)							# Flux
			J[msk] = getJ(u[msk,k],uxm[msk]); J[~msk] = getJ(up[~msk],uxp[~msk]);		# Get flux from right or left according to msk
			u[:,k+1] = u[:,k] - dt*(Dx1(J,dx,1)) - dt*eta*(up2-4.*up+6.*u[:,k]-4.*um+um2)/(dx**4)		# Update u
	
	# For the next bunch, it turns out that if you conditionally difference u and NOT ux, the results are
	# much better than if you conditionally difference both of them.  I've included both options for 
	# comparison.  Schemes 3 and 5 conditionally difference both u and ux, while  4 and 6 do only u. 
	
	# The idea of each of the following is identical to scheme 2, so I'll forgo excessive commenting. 
	elif scheme==3:		# Small gradient-norm differencing 1: condition u and ux
		for k in range(nt-1):
			um = roll(u[:,k],1); up = roll(u[:,k],-1);
			um2 = roll(um,1); up2 = roll(up,-1)
			uxm = (u[:,k]-um)/dx; uxp = (up2-up)/dx;
			msk = abs(up2-up) > abs(u[:,k]-um)
			J = np.empty(up.shape)
			J[msk] = getJ(u[msk,k],uxm[msk]); J[~msk] = getJ(up[~msk],uxp[~msk]);
			u[:,k+1] = u[:,k] - dt*(Dx1(J,dx,1)) - dt*eta*(up2-4.*up+6.*u[:,k]-4.*um+um2)/(dx**4)
	elif scheme==4:		# Small gradient-norm differencing 2: condition u only
		for k in range(nt-1):
			um = roll(u[:,k],1); up = roll(u[:,k],-1);
			um2 = roll(um,1); up2 = roll(up,-1)
			uxm = (u[:,k]-um)/dx; uxp = (up2-up)/dx; ux = (up-u[:,k])/dx;
			msk = abs(up2-up) > abs(u[:,k]-um)
			J = np.empty(up.shape)
			J[msk] = getJ(u[msk,k],ux[msk]); J[~msk] = getJ(up[~msk],ux[~msk]);
			u[:,k+1] = u[:,k] - dt*(Dx1(J,dx,1)) - dt*eta*(up2-4.*up+6.*u[:,k]-4.*um+um2)/(dx**4)
	elif scheme==5:		# Upwind differencing 1: condition u and ux
		for k in range(nt-1):
			um = roll(u[:,k],1); up = roll(u[:,k],-1);
			um2 = roll(um,1); up2 = roll(up,-1)
			uxp = (up2-up)/dx; uxm = (u[:,k]-um)/dx;
			J = getJ(up,uxp); Jm = getJ(u[:,k],uxm)
			msk = np.sign(up-um)*np.sign(-Dx1(J+Jm,dx,1)) == -1
			J[msk] = Jm[msk]
			u[:,k+1] = u[:,k] - dt*(Dx1(J,dx,1)) - dt*eta*(up2-4.*up+6.*u[:,k]-4.*um+um2)/(dx**4)
	elif scheme==6:		# Upwind differencing 2: condition u only
		for k in range(nt-1):
			um = roll(u[:,k],1); up = roll(u[:,k],-1);
			um2 = roll(um,1); up2 = roll(up,-1)
			uxp = (up2-up)/dx; uxm = (u[:,k]-um)/dx; ux = (up-u[:,k])/dx;
			J = getJ(up,ux); Jm = getJ(u[:,k],ux)
			msk = np.sign(up-um)*np.sign(-Dx1(J+Jm,dx,1)) == -1
			J[msk] = Jm[msk]
			u[:,k+1] = u[:,k] - dt*(Dx1(J,dx,1)) - dt*eta*(up2-4.*up+6.*u[:,k]-4.*um+um2)/(dx**4)
	
	if plots==1:
		pylab.plot(x,u[:,::1000]); pylab.show()
	elif plots==2:
		pylab.plot(u[:,-1]);pylab.show();
	
	return u
