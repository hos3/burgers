# Animates evolution of solution u given by burgerses.py
	
from matplotlib import animation
import pylab as pl
import numpy as np

def animate(u,fname,lx=0.0,hx=2.0*np.pi,ly=0.0,hy=1.0):
	# Animates evolution of u, optionally saving the result in file "fname"
	# lx, hx, ly, and hy are bounds for the animation box
	
	fig = pl.figure()
	ax = pl.axes(xlim = (lx,hx), ylim = (ly,hy))
	line, = pl.plot([],[],lw=2)
	
	def init():
		line.set_data([],[])
		return line,
	
	def animstep(i):
		x = np.linspace(0,2*np.pi,u.shape[0])
		y = u[:,i*100]
		line.set_data(x,y)
		return line,
	
	anim = animation.FuncAnimation(fig, animstep, init_func=init, frames=int(np.floor(u.shape[1]/100.)), interval=20, blit=True)
	
	if input('Enter any value to save. '):	anim.save(fname)
