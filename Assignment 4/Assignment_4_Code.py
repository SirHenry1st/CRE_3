#%%
# Import
import numpy as np
import scipy.integrate as integ 
from scipy.integrate import solve_bvp 
import matplotlib.pyplot as plt 

#%%

t = np.linspace(1,100,1000) # time in s
t_bar = 25 #mean residence time in s

# Calculating F(t) for laminar flow
F_lam = 1-0.25*(t_bar/t)**2
# Replacing all values below 0
F_lam = np.where(F_lam<0, 0, F_lam)

# Calculating F(t) for plug flow
F_plug = np.where(t<25,0,t)
F_plug = np.where(F_plug>25,1,F_plug)

# Plotting
fig = plt.figure(figsize=(3, 3), dpi=150)
plt.plot(t, F_lam, color="red",  linewidth=1.0, linestyle="-")
plt.plot(t, F_plug, color="blue",  linewidth=1.0, linestyle="-")
plt.xlim(0, t.max())
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=10, rotation=0)
plt.yticks(fontsize=10, rotation=0)
plt.xlabel('time / s',fontsize=10)
plt.ylabel('$F(t)$ / 1',fontsize=10)
plt.title('Cumulative distribution functions',fontsize=10)
plt.plot(t, F_lam, color="red",  linewidth=1.0, linestyle="-", label="laminar flow")
plt.plot(t, F_plug, color="blue",  linewidth=1.0, linestyle="-", label="plug flow")
plt.legend(loc='lower right', prop={'size': 10}, frameon=False)
plt.show()

#%%
t = np.linspace(1,50,1000) # time in s
t_bar = 25 #mean residence time in s

# Calculating E(t) for different Bo
E_100 = (1/(4*np.pi/100)**0.5)*np.exp(-(1-(t/t_bar))**2/(4/100))
E_1000 = (1/(4*np.pi/1000)**0.5)*np.exp(-(1-(t/t_bar))**2/(4/1000))
E_10000 = (1/(4*np.pi/10000)**0.5)*np.exp(-(1-(t/t_bar))**2/(4/10000))

# Plotting
fig = plt.figure(figsize=(5,3), dpi=150)
plt.plot(t/t_bar, E_100, color="red",  linewidth=1.0, linestyle="-")
plt.plot(t/t_bar, E_1000, color="blue",  linewidth=1.0, linestyle="-")
plt.plot(t/t_bar, E_10000, color="green",  linewidth=1.0, linestyle="-")
plt.xlim(0.5, 1.5)
plt.ylim(-0.05, E_10000.max()*1.1)
plt.xticks(fontsize=10, rotation=0)
plt.yticks(fontsize=10, rotation=0)
plt.xlabel('$\Theta$ / 1',fontsize=10)
plt.ylabel('$E(\Theta)$ / s$^-1$',fontsize=10)
plt.title('Residence time distributions for different $Bo$-numbers',fontsize=10)
plt.plot(t/t_bar, E_100, color="red",  linewidth=1.0, linestyle="-", label="$Bo=100$")
plt.plot(t/t_bar, E_1000, color="blue",  linewidth=1.0, linestyle="-", label="$Bo=1000$")
plt.plot(t/t_bar, E_10000, color="green",  linewidth=1.0, linestyle="-", label="$Bo=10000$")
plt.legend(loc='upper right', prop={'size': 10}, frameon=False)
plt.show()

#%%
#Parameters
D_Aeff = 1E-6 # effective diffusion coefficient m^2/s
k_1 = 0.01 # reaction rate constant 1/s
k_fl = 1 # outer mass transfer coefficient m/s
c_Abulk = 1 # concentration in bulk mol/m^3
chi = 0.005 # spatial length of the catalyst m

#BVP-solver
def fun_bvp_flat(x, y):
  c_A = y[0] #transferring to local variable
  dc_Adx = y[1] #transferring to local variable
  x_a = x

  dydx = dc_Adx # y''
  d2ydx2 = (k_1/D_Aeff)*c_A -(0/(x_a))*y[1] #y'

  sol = np.vstack((dydx, d2ydx2)) #stack the results

  return sol #return solution

def fun_bvp_cylinder(x,y):
  c_A = y[0] #transferring to local variable
  dc_Adx = y[1] #transferring to local variable
  x_a = x

  dydx = dc_Adx # y''
  d2ydx2 = (k_1/D_Aeff)*c_A -(1/(x_a))*y[1] #y'

  sol = np.vstack((dydx, d2ydx2)) #stack the results

  return sol #return solution

def fun_bvp_sphere(x,y):
  c_A = y[0] #transferring to local variable
  dc_Adx = y[1] #transferring to local variable
  x_a = x

  dydx = dc_Adx # y''
  d2ydx2 = (k_1/D_Aeff)*c_A -(2/(x_a))*y[1] #y'

  sol = np.vstack((dydx, d2ydx2)) #stack the results

  return sol #return solution

# Boundary condition (bc)
def bc(ya, yb):
  bc_x0 = ya[1] #bc @ pos x=0 ya[1]=dc_Adx
  bc_xchi = yb[1] + k_fl/D_Aeff*(yb[0]-c_Abulk) #bc @ pos x=0.01m yb[0]=c_A and yb[1]=dc_Adx
  return np.array([ bc_x0, bc_xchi]) #return solution

x = np.linspace(chi/10000, chi, 100) #defining range. Please note, lower value is chi/10000, because x=0 is not possible
y_a_init = np.zeros((2, x.size))
y_a = y_a_init
# Call solver
sol_flat = solve_bvp(fun_bvp_flat, bc, x, y_a)
sol_cylinder = solve_bvp(fun_bvp_cylinder, bc, x, y_a)
sol_sphere = solve_bvp(fun_bvp_sphere, bc, x, y_a)

# Success of solver and number of iterations
print(sol_flat.success, sol_cylinder.success, sol_sphere.success) # solver successful
print('Number of itarations: ', sol_flat.niter, sol_cylinder.niter, sol_sphere.niter)

# Plot
fig = plt.figure(figsize=(5,3), dpi=150)
plt.plot(sol_flat.x, sol_flat.y[0,:], label="flat geometry")
plt.plot(sol_cylinder.x, sol_cylinder.y[0,:], label="cyclindrical geometry")
plt.plot(sol_sphere.x, sol_sphere.y[0,:], label="spherical geometry")
plt.legend()
plt.xlabel("$\chi$ / m")
plt.ylabel("$c_A$ / mol m$^-3$")
plt.show()

