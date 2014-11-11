import scipy, pylab

N = 200
x = scipy.arange(0.,2.*scipy.pi,2.*scipy.pi/N)
dx = x[1]-x[0]
shift = 0.1
u0 = scipy.sin(x) + shift

def ddx(f):
    return (scipy.roll(f,1)-scipy.roll(f,-1))/(2.*dx)

""" Doesn't work: want big dudx for delta shock
def dfdxUpwind(f,u):
    return (u<=0)*(f-scipy.roll(f,-1))/dx + (u>0)*(scipy.roll(f,1)-f)/dx
"""

def d2dx2(f):
    return (scipy.roll(f,1)-2*f+scipy.roll(f,-1))/dx**2

def dudt(u,eps,eta):
    J = u*u/(1+eps*(ddx(u))**2)
    return 0.5*ddx(J) + eta * d2dx2(u)

def Euler(u0, eta=0.005, eps=0.005, T=1.0, dt = 0.0001):
   u = u0.copy()
   for t in scipy.arange(0.,T,dt):
       u += dt*dudt(u,eps,eta)
   return u

pylab.clf()
pylab.plot(x,Euler(u0,eta=0.005,eps=0.005,T=40.,dt=0.0004))
pylab.show()
