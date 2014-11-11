'''Solves a 2D Burgers' equation in a variety of ways.'''
from keyboard import keyboard 		# For debugging
import numpy as np; import scipy as sp; import pylab; from scipy import fftpack; from copy import copy;
from mpl_toolkits.mplot3d import art3d

roll = np.roll							# Used often
pi = np.pi

# Default values:
eps0 = .001								# epsilon
eta0 = .005								# eta (viscosity)
nx0 = 100								# number of x grid points
nt0 = 5000								# number of time grid points
dx0 = 1./nx0							# x grid spacing
dt0 = .0004								# time grid spacing
u_init0 = lambda t,s: (1-np.cos(2*pi*t))*(1-np.cos(2*pi*s)) - .9		# Initial distribution

def solve(nx=nx0, nt=nt0, dx=None, dt=dt0, u_init=u_init0, eps=eps0, eta=eta0, flux=0, scheme=1, ordr=2, plots=1,shift=0.):
	'''Solves the 2D Burgers' equation.'''
	
	if not np.isscalar(nx):		# Allow (nx,ny) to be specified as a tuple
		ny = nx[1]; nx = nx[0];
	else:
		ny = copy(nx)
	
	if dx==None:				# A common default setting
		dx = 1./nx; dy = 1./ny;	
	elif not np.isscalar(dx):		# Allow (dx,dy) to be specified as a tuple
		dy = dx[1]; dx = dx[0];
	else:
		dy = copy(dx)
	
	if not (type(nx) is int):	# Allow either distance or number of steps to be specified
		dx = nx/np.ceil(nx/dx)
		nx = int(np.round(nx/dx))
	if not (type(ny) is int):
		dy = ny/np.ceil(ny/dy)
		ny = int(np.round(ny/dy))
	if not (type(nt) is int):
		dt = nt/np.ceil(nt/dt)
		nt = int(np.round(nt/dt))
	
	# Make some differentiation functions:
	Dx = lambda u: (roll(u,-1,0)-roll(u,1,0))/2/dx
	Dx1 = lambda u,pm: (u-roll(u,pm,0))/dx/pm
	Dy = lambda u: (roll(u,-1,1)-roll(u,1,1))/2/dy
	Dy1 = lambda u,pm: (u-roll(u,pm,1))/dy/pm
	
	def fsmooth(v):
		freqx = fftpack.fftfreq(v.shape[0])/dx
		freqy = fftpack.fftfreq(v.shape[1])/dy
		freqx, freqy = np.meshgrid(freqx,freqy)
		return sp.real(fftpack.ifft2( \
			fftpack.fft2(v)*sp.exp(eta*(2j*pi)**ordr*(freqx**ordr + freqy**ordr) * dt) ))
	
	x = np.arange(nx)*dx; y = np.arange(ny)*dy; xgrid,ygrid = np.meshgrid(x,y);
	u = np.empty((nx,ny,nt))		# Solution array
	#Js = np.empty((nx,nt,nt-1))	# Fluxes
	
	if callable(u_init):
		u[:,:,0] = u_init(xgrid,ygrid) + shift
	else:
		u[:,:,0] = u_init + shift
	
	if flux==0:
		getJ = lambda u,ux,uy: u**2 / 2.
	elif flux==1:
		getJ = lambda u,ux,uy: u**2 * sp.exp(-eps*(ux**2+uy**2)) /2.
	elif flux==2:
		getJ = lambda u,ux,uy: u**2 / 2. / (1+eps*(ux**2+uy**2))
		
	if scheme==1:
		for k in range(nt-1):
			u0 = u[:,:,k]; 
			uxc = Dx(u0); uyc = Dy(u0);						# uxc stands for "du/dx, centered"
			J = getJ(u0,uxc,uyc); #Js[:,:,k] = J
			u[:,:,k+1] = fsmooth(u0 + dt*(Dx(J)+Dy(J)))		# fsmooth is not conserving mass...
			
	elif scheme==2:			# Downhill differencing 1
		for k in range(nt-1):
			u0 = u[:,:,k]
			ux = Dx1(u0,-1); uxc = Dx(u0);
			uy = Dy1(u0,-1); uyc = Dy(u0);
			Jx = np.empty(u0.shape); Jy = np.empty(u0.shape);
			mskx = abs(u0)<=abs(roll(u0,-1,0)); msky = abs(u0)<=abs(roll(u0,-1,1))
			Jx[mskx] = getJ(u0[mskx],ux[mskx],.5*(uyc[mskx] + (roll(uyc,-1,0))[mskx]))
			Jx[~mskx] = getJ((roll(u0,-1,0))[~mskx],ux[~mskx],.5*(uyc[~mskx] + (roll(uyc,-1,0))[~mskx]))
			Jy[msky] = getJ(u0[msky],.5*(uxc[msky] + (roll(uxc,-1,1))[msky]),uy[msky])
			Jy[~msky] = getJ((roll(u0,-1,1))[~msky],.5*(uxc[~msky]+(roll(uxc,-1,1))[~msky]),uy[~msky])
			
			u[:,:,k+1] = fsmooth(u0 + dt*(Dx1(Jx,1) + Dy1(Jy,1)))
	elif scheme==3:			# Upwind differencing 1
		for k in range(nt-1):
			u0 = u[:,:,k]
			ux = Dx1(u0,-1); uxc = Dx(u0);
			uy = Dy1(u0,-1); uyc = Dy(u0);
			J = getJ(u0,uxc,uyc); Jx = np.empty(u0.shape); Jy = np.empty(u0.shape);
			mskx = np.sign(Dx(J))*np.sign(uxc)==1; msky = np.sign(Dy(J))*np.sign(uyc)==1
			Jx[mskx] = getJ(u0[mskx],ux[mskx],.5*(uyc[mskx] + (roll(uyc,-1,0))[mskx]))
			Jx[~mskx] = getJ((roll(u0,-1,0))[~mskx],ux[~mskx],.5*(uyc[~mskx] + (roll(uyc,-1,0))[~mskx]))
			Jy[msky] = getJ(u0[msky],.5*(uxc[msky] + (roll(uxc,-1,1))[msky]),uy[msky])
			Jy[~msky] = getJ((roll(u0,-1,1))[~msky],.5*(uxc[~msky]+(roll(uxc,-1,1))[~msky]),uy[~msky])
			
			u[:,:,k+1] = fsmooth(u0 + dt*(Dx1(Jx,1) + Dy1(Jy,1)))
	
	if plots==1: plotu(u,xgrid,ygrid)
	if type(plots)==type('animate'): 			# Animate the evolution of u.  'plots' contains the file name for saving the animation.
		import surfAnimate
		print('Animating evolution.  This could take some time... ')
		surfAnimate.anim(u,xgrid,ygrid,25,rstride=u.shape[0]//50,cstride=u.shape[1]//50,retrn=0,noshow=plots)
	
	return u


def plotu(u,x,y,Tidx=-1,figr=1):
	if x==None:
		x=np.arange(u.shape[0]); y=np.arange(u.shape[1])
		x,y = np.meshgrid(x,y)
	if figr==1:
		fig = pylab.figure()
	elif figr=='over':
		fig = pylab.gcf()
		fig.clear()
	ax = fig.add_subplot(111,projection='3d')
	ax.plot_surface(x,y,u[:,:,Tidx]); pylab.show();