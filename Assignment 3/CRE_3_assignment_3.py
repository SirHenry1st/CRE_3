#%%
# Import

import numpy as np
import scipy.integrate as integ  # important for initial value problem
import matplotlib.pyplot as plt  # figures
plt.style.use("ICIWstyle")

#%%
# Data Parameters

E_A = 72500      # activation energy J/mol
k_0 = 5.01E06       # pre-exponential factor 1/s
R = 8.314        # gas constant J/(mol K)
H_R = -92200     # reaction enthalpy J/mol

d_R = 0.03       # tube diameter m
L_R = 300        # reactor length m
h_wall = 500     # wall heat transfer coefficient W/(m^2 K)  [= 0.5 kJ/(s m^2 K)]
u = 1.0          # fluid velocity m/s

rho = 1000       # density of reaction mixture kg/m^3
c_p = 4190       # heat capacity J/(kg K)

# Initial conditions (= inlet conditions for PFTR)
c_10 = 2270      # inlet concentration EO mol/m^3
c_20 = 0         # inlet concentration EG mol/m^3
T_0 = 423        # inlet temperature K
T_wall = 423     # wall temperature K (fixed, independent of position)

# Precalculations
C = c_p * rho              # volumetric heat capacity J/(m^3 K)
a_wall = 4 / d_R           # specific wall area m^2_wall/m^3_reactor  (= pi*d / (pi*d^2/4))

def kinetics(T, c_1):
    """Arrhenius reaction rate, first order in c_1."""
    k_1 = k_0 * np.exp(-E_A / (R * T))   # rate constant 1/s
    r = k_1 * c_1                          # reaction rate mol/(m^3 s)
    return r

def PFTR_constant_h(L, f):
    """
    Spatial derivatives for the PFTR material and energy balances.

    f[0] = c_1  (EO concentration, mol/m^3)
    f[1] = c_2  (EG concentration, mol/m^3)
    f[2] = T    (temperature, K)

    Space-time transformation:  d/dL = (1/u) * d/dt
    Each right-hand side is the bSTR time-derivative divided by u.
    """
    c_1 = f[0]
    c_2 = f[1]
    T   = f[2]

    r = kinetics(T, c_1)

    # Component balances (material balance per unit reactor length)
    dc_1dL = -r / u
    dc_2dL =  r / u

    # Energy balance per unit reactor length
    # Reaction term:    -H_R * r / C              [K/s] / [m/s] = K/m
    # Heat-transfer:    h_wall * a_wall * (T-T_wall) / C  [K/m]
    dTdL = (-H_R * r / C - h_wall * a_wall * (T - T_wall) / C) / u

    dfdL = np.empty_like(f)
    dfdL[0] = dc_1dL
    dfdL[1] = dc_2dL
    dfdL[2] = dTdL
    return dfdL

# Initial (inlet) conditions
f_init = np.array([c_10, c_20, T_0])

# Solve
Lspan = np.array([0, L_R])
leval = np.linspace(0, Lspan[1], 2001)
sol = integ.solve_ivp(PFTR_constant_h, Lspan, f_init, method='BDF', t_eval=leval)

# POSTPROCESSING & PLOTTING

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.set(xlabel='$L$ / m', ylabel='$c_i$ / mol L$^{-1}$')
ax1.plot(sol.t, sol.y[0, :] / 1000, 'b-',  label='$c_{\\mathrm{EO}}$')
ax1.plot(sol.t, sol.y[1, :] / 1000, 'b--', label='$c_{\\mathrm{EG}}$')
ax1.legend()

ax2.set(xlabel='$L$ / m', ylabel='$T$ / K')
ax2.plot(sol.t, sol.y[2, :], 'r-')

fig.tight_layout()
plt.show()

#%%

def PFTR(L, f, h, T_in):
    c_1, c_2, T = f[0], f[1], f[2]
    r = kinetics(T, c_1)
    dc_1dL = -r / u
    dc_2dL =  r / u
    dTdL = (-H_R * r / C - h * a_wall * (T - T_wall) / C) / u
    dfdL = np.empty_like(f)
    dfdL[0] = dc_1dL
    dfdL[1] = dc_2dL
    dfdL[2] = dTdL
    return dfdL

Lspan = np.array([0, L_R])
leval = np.linspace(0, Lspan[1], 2001)

#%%
# Sensitivity 1: Cooling Power (h_wall)

h_cases = {
    'Adiabatic ($h = 0$)':                   0,
    '$h = 50$ W m$^{-2}$ K$^{-1}$':         50,
    '$h = 500$ W m$^{-2}$ K$^{-1}$ (base)': 500,
    '$h = 5000$ W m$^{-2}$ K$^{-1}$':       5000,
    'Isothermal ($h \\to \\infty$)':         1e6,
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.set(xlabel='$L$ / m', ylabel='$c_{\\mathrm{EO}}$ / mol L$^{-1}$')
ax2.set(xlabel='$L$ / m', ylabel='$T$ / K')
colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:cyan', 'tab:green']

for (label, h), color in zip(h_cases.items(), colors):
    f_init = np.array([c_10, c_20, T_0])
    sol = integ.solve_ivp(PFTR, Lspan, f_init, args=(h, T_0), method='BDF', t_eval=leval)
    ax1.plot(sol.t, sol.y[0, :] / 1000, color=color, label=label)
    ax2.plot(sol.t, sol.y[2, :],         color=color, label=label)

ax1.legend(fontsize=8)
ax2.legend(fontsize=8)
fig.tight_layout()
plt.show()

#%%
# Sensitivity 2: Inlet Concentration (base: h=500, T_0=423 K)

c_cases = {
    '$c_{\\mathrm{EO,0}} = 1$ mol L$^{-1}$':  1000,
    '$c_{\\mathrm{EO,0}} = 2$ mol L$^{-1}$':  2000,
    '$c_{\\mathrm{EO,0}} = 5$ mol L$^{-1}$':  5000,
    '$c_{\\mathrm{EO,0}} = 10$ mol L$^{-1}$': 10000,
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.set(xlabel='$L$ / m', ylabel='$c_{\\mathrm{EO}}$ / mol L$^{-1}$')
ax2.set(xlabel='$L$ / m', ylabel='$T$ / K')
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

for (label, c_in), color in zip(c_cases.items(), colors):
    f_init = np.array([c_in, c_20, T_0])
    sol = integ.solve_ivp(PFTR, Lspan, f_init, args=(h_wall, T_0), method='BDF', t_eval=leval)
    ax1.plot(sol.t, sol.y[0, :] / 1000, color=color, label=label)
    ax2.plot(sol.t, sol.y[2, :],         color=color, label=label)

ax1.legend(fontsize=8)
ax2.legend(fontsize=8)
fig.tight_layout()
plt.show()

#%%
# Sensitivity 3: Inlet Temperature (base: h=500, c_10=2270 mol/m^3)
# 150°C = 423 K, 200°C = 473 K, 250°C = 523 K

T_cases = {
    '$T_0 = 150$ °C (423 K)': 423,
    '$T_0 = 200$ °C (473 K)': 473,
    '$T_0 = 250$ °C (523 K)': 523,
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.set(xlabel='$L$ / m', ylabel='$c_{\\mathrm{EO}}$ / mol L$^{-1}$')
ax2.set(xlabel='$L$ / m', ylabel='$T$ / K')
colors = ['tab:blue', 'tab:orange', 'tab:red']

for (label, T_in), color in zip(T_cases.items(), colors):
    f_init = np.array([c_10, c_20, T_in])
    sol = integ.solve_ivp(PFTR, Lspan, f_init, args=(h_wall, T_in), method='BDF', t_eval=leval)
    ax1.plot(sol.t, sol.y[0, :] / 1000, color=color, label=label)
    ax2.plot(sol.t, sol.y[2, :],         color=color, label=label)

ax1.legend(fontsize=8)
ax2.legend(fontsize=8)
fig.tight_layout()
plt.show()

#%%
