'''Animates evolution of a surface.'''

import numpy as np; import scipy as sp; import pylab as pl
from matplotlib import animation; from mpl_toolkits.mplot3d.axes3d import Axes3D;

def anim(z,x=None,y=None,Tstride=1,start=0,end=np.inf,rstride=1,cstride=1,\
	cmap='jet',linewidth=0.,interval=50,fps=30,noshow=0,retrn=1):
	'''Animates evolution of a 3D array z, assuming that the third dimension represents a time parameter.
		x and y are coordinates against which to plot z; by default, these are made to be indices.
		The animation can be made from slices of z by specifying 'Tstride', 'start', and 'end', 
		so that the subarray z[:,:,start:end:Tstride] is the effective array used.
		noshow==0 means show the animation and prompt for saving;
		noshow==1 means don't show the animation (presumably just return it as function output)
		noshow==<string> indicates that the animation should just be saved and not shown in a GUI;
		in this case, the value of noshow is taken as the file name under which to save.
		If retrn==1, the animation object is returned; otherwise nothing is returned.
		Other parameters are passed as inputs to animation.FuncAnimate or the animation saver.'''
	
	# Allow some convenience inputs
	if end==np.inf: end = z.shape[2]
	if x == None: x = np.arange(z.shape[0])
	if len(x.shape)==1: x = np.tile(x,(z.shape[0],1))
	if y == None: y = np.arange(z.shape[1])
	if len(y.shape)==1: y = np.tile(y,(z.shape[1],1)).T
	
	fig = pl.figure()							# Figure
	ax = fig.add_subplot(111,projection='3d')	# 3D axes
	s = ax.plot_surface([],[],[])				# Initialize surface object
	mx = np.amax(z[:,:,0]); mn = np.amin(z[:,:,0])		# Max and min, for setting axes scales
	if (mx==np.nan) or (mx==np.inf): np.median(np.unique(np.abs(np.nan_to_num(z[:,:,0]))))	# In case of nan or inf...
	if (mn==np.nan) or (mn==np.inf): -np.median(np.unique(np.abs(np.nan_to_num(z[:,:,0]))))
	
	def a_init():								# Initialization function for animation
		ax.clear()
		ax.set_zlim(mn,mx)
		s = ax.plot_surface([],[],[])
		return s,
	
	def a_update(t):							# Now plot actual data for a fixed time
		ax.clear()
		ax.set_zlim(mn,mx)
		s = ax.plot_surface(x,y,z[:,:,t],\
			rstride=rstride,cstride=cstride,cmap=cmap,linewidth=linewidth)
		return s,
	
	anim = animation.FuncAnimation(fig,a_update,range(start,end,Tstride),a_init,interval=interval)
	
	if noshow==0: 
		pl.show()
		sav = input('Save animation? (Press enter for no, or type a filename with extension to save.) ')
		if sav: anim.save(sav,fps=fps)
	elif noshow==1:
		pass
	else:
		anim.save(noshow,fps=fps)
		pl.close(fig)
	
	if retrn: return anim
