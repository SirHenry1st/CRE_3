## 1) Import of all required modules
import numpy as np
from scipy.integrate import solve_ivp # sub-package integrate from scipy
import matplotlib.pyplot as plt
## 2) Definition of parameters
# values in SI units!
pi = np.pi
A = 1 # m^2, surface
m = 10 # kg, mass
cp = 2000 # J/kgK, heat capacity
T_Env = 20+273.15 # K, temperature of the environment
alpha = 20 # W/m^2K, heat transfer coefficient
tEnd = 2*3600 # s, end time
T_0 = np.array([100+273.15]) # K, Initial temperature (as array required by solver)
## 3) Definition of functions
def coolingCurve(t,T): # arguments t = time and T = temperature
    Qdot = alpha*A*(T-T_Env) # algebraic equation
    dTdt = - Qdot/cp/m # ordinary differential equations
    return dTdt # solution of the system of equations from lines 22 and 24
## 4) Configuration and call of solvers
tSpan = (0,tEnd) # tupel with initial and end time
tSteps = np.linspace(tSpan[0],tSpan[1],101)
# simulation results are saved in the vaiable sol
sol = solve_ivp(coolingCurve,tSpan,T_0,t_eval=tSteps)
# Assignemnt of simulation results separately
T = sol.y[0] # temperature as property y in sol
t = sol.t # time steps as property t in sol
## 5) Graphical representation of simulation results
fig, ax = plt.subplots() # create blank figure
ax.plot(t/3600,T-273.15) # plot values (Units!)
ax.set(ylabel='$T$ / °C')
ax.set(xlabel='$t$ / h')
plt.show