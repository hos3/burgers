'''burgerses.py
Solves modified Burgers' equations by a variety of schemes.  This version employs a Fourier method for 
artificial viscosity, and allows specification of arbitrary order for the viscosity. 
This version differs from the other FourierBurgerses in that it takes the total time and spatial extents instead of the number of time steps/grid points (it can also return differently).  This makes it easier to look at convergence properties.
'''

import scipy as np						# Started out using numpy, then switched to scipy for consistency with expdelta
import scipy as sp
import pylab
from scipy import fftpack

roll = np.roll							# Used often

# Default values:
eps0 = .00005							# epsilon
eta0 = .00001							# eta (viscosity)
X00 = 1.								# number of x grid points
T00 = .5								# time to run for
dx0 = 1./200.							# x grid spacing
dt0 = .00004							# time grid spacing
u_init0 = lambda x: .2+np.sin(2.*np.pi*x/X00)		# Initial distribution

def Dx(f,dx):							# Symmetric derivative
	return (roll(f,-1) - roll(f,1))/(2.*dx)

def Dx1(f,dx,pm):						# Asymmetric derivative: pm==-1 differences upwards, pm==1 differences downwards
	return (f-roll(f,pm))/(pm*dx)	

def solve(X0=X00, T0=T00, dx=dx0, dt=dt0, u_init=u_init0, eps=eps0, eta=eta0, flux=2, scheme=6, ordr=2, plots=1,last=0,vb=0):
	'''Solves du/dt = -dJ/dx + eta * uxx.
		Returns (u,JS), where u is a shape (nx,nt) array with u[j,k] = the solution at x=j*dx & t=k*dt,
			and JS is a shape (nx,nt-1) array with JS[j,k] = the flux at x=(j+1/2)*dx & t=k*dt.
		J is set via the flux parameter; flux==0 gives J==.5*u**2, flux==1 gives J==.5*u**2*exp(-eps*ux**2),
			and flux==2 gives J==.5*u**2/(1.+eps*ux**2); if flux is a function, J = flux
		A variety of differencing schemes are available, via the scheme parameter.	
		The plots keyword determines whether to plot nothing (plots==0), the full time evolution (plots==1)
		or just the final state (plots==2).
		If 'last' is set to 1, then only the final state is returned instead of the entire arrays u & JS.
		If 'vb' is set to 1, then the code prints status updates.
		'''
	
	nx = int(np.floor(X0/dx)); dx = X0/nx; nt = int(np.floor(T0/dt)); dt = T0/nt;
	
	def fsmooth(u):
		'''Smooths data u by evolving it through a time step of a diffusion equation.
		The diffusion equation may be second or fourth (or higher) order, depending on the 
		input 'ordr'.  The diffusion constant is eta and the time step is dt.
		Smoothing is done in Fourier space for stability.'''
		freq = fftpack.fftfreq(len(u))
		Fu = fftpack.fft(u)*sp.exp(eta*dt*(2j*np.pi*freq/dx)**ordr)
		return sp.real(fftpack.ifft(Fu))
	
	u = np.empty((nx,nt))
	x = np.arange(nx)*dx
	if callable(u_init):
		u[:,0] = u_init(x)				# Initial conditions
	else:
		u[:,0] = u_init
	
	JS = np.empty((nx,nt-1))
	
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
			ux = Dx(u[:,k],dx)								#...for "minus 1", in the context of index increments
			J = getJ(u[:,k],ux)
			JS[:,k] = J
			u[:,k+1] = fsmooth(u[:,k] - dt*Dx(J,dx))	# Update u (with smoothing from artificial viscosity
			
	elif scheme==2:		# Absolute downhill differencing
		for k in range(nt-1):
			um = roll(u[:,k],1); up = roll(u[:,k],-1); up2 = roll(up,-1);
			uxm = (u[:,k]-um)/dx; uxp = (up2-up)/dx;		# Left and right derivatives
			msk = abs(up) > abs(u[:,k])						# Determine which way to difference
			J = np.empty(up.shape)							# Flux
			J[msk] = getJ(u[msk,k],uxm[msk]); J[~msk] = getJ(up[~msk],uxp[~msk]);		# Get flux from right or left according to msk
			u[:,k+1] = fsmooth(u[:,k] - dt*(Dx1(J,dx,1)))		# Update u
	
	# For the next bunch, it turns out that if you conditionally difference u and NOT ux, the results are
	# much better than if you conditionally difference both of them.  I've included both options for 
	# comparison.  Schemes 3 and 5 conditionally difference both u and ux, while  4 and 6 do only u. 
	
	# The idea of each of the following is identical to scheme 2, so I'll forgo excessive commenting. 
	elif scheme==3:		# Small gradient-norm differencing 1: condition u and ux
		for k in range(nt-1):
			um = roll(u[:,k],1); up = roll(u[:,k],-1); up2 = roll(up,-1);
			uxm = (u[:,k]-um)/dx; uxp = (up2-up)/dx;
			msk = abs(up2-up) > abs(u[:,k]-um)
			J = np.empty(up.shape)
			J[msk] = getJ(u[msk,k],uxm[msk]); J[~msk] = getJ(up[~msk],uxp[~msk]);
			u[:,k+1] = fsmooth(u[:,k] - dt*(Dx1(J,dx,1)))
	elif scheme==4:		# Small gradient-norm differencing 2: condition u only
		for k in range(nt-1):
			um = roll(u[:,k],1); up = roll(u[:,k],-1); up2 = roll(up,-1);
			uxm = (u[:,k]-um)/dx; uxp = (up2-up)/dx; ux = (up-u[:,k])/dx;
			msk = abs(up2-up) > abs(u[:,k]-um)
			J = np.empty(up.shape)
			J[msk] = getJ(u[msk,k],ux[msk]); J[~msk] = getJ(up[~msk],ux[~msk]);
			u[:,k+1] = fsmooth(u[:,k] - dt*(Dx1(J,dx,1)))
	elif scheme==5:		# Upwind differencing 1: condition u and ux
		for k in range(nt-1):
			um = roll(u[:,k],1); up = roll(u[:,k],-1); up2 = roll(up,-1);
			uxp = (up2-up)/dx; uxm = (u[:,k]-um)/dx;
			J = getJ(up,uxp); Jm = getJ(u[:,k],uxm)
			msk = np.sign(up-um)*np.sign(-Dx1(J+Jm,dx,1)) == -1
			J[msk] = Jm[msk]
			u[:,k+1] = fsmooth(u[:,k] - dt*(Dx1(J,dx,1)))
	elif scheme==6:		# Upwind differencing 2: condition u only
		for k in range(nt-1):
			um = roll(u[:,k],1); up = roll(u[:,k],-1); up2 = roll(up,-1);
			uxp = (up2-up)/dx; uxm = (u[:,k]-um)/dx; ux = (up-u[:,k])/dx;
			J = getJ(up,ux); Jm = getJ(u[:,k],ux)
			msk = np.sign(up-um)*np.sign(-Dx1(J+Jm,dx,1)) == -1
			J[msk] = Jm[msk]
			JS[:,k] = J
			u[:,k+1] = fsmooth(u[:,k] - dt*(Dx1(J,dx,1)))
	
	if plots==1:
		pylab.figure()
		pylab.plot(x,u[:,::1000]); pylab.show()
	elif plots==2:
		pylab.figure()
		pylab.plot(u[:,-1]);pylab.show();
	
	if not last:
		return u,JS
	else:
		return u[:,-1],JS[:,-1]


def plotu(u,tstep=1000,start=0,end=np.inf,figr=1):
	if end==np.inf: end = u.shape[1]
	if figr: pylab.figure();
	pylab.plot(np.linspace(0,2*np.pi,u.shape[0]),u[:,start:end:tstep]); pylab.show();