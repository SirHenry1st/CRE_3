#%%
# Import

import numpy as np
from scipy.interpolate import interp1d
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
A_ref = np.pi * d_R ** 2 / 4

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
M_WA = 18.01 # Molar mass of water in g/mol
M_EG = 62.07 # Molar mass of ethylene glycol in g/mol
M_EO = 44.05 # Molar mass of ethylene oxide in g/mol

R_WA = 8.314 / M_WA # Specific heat capacity of water in 
R_EG = 8.314 / M_EG # Specific heat capacity of ethylene glycol in 
R_EO = 8.314 / M_EO # Specific heat capacity of ethylene oxide in 

T_C_EG = 724.05 # Critical temperature of ethylene glycol in K
T_C_WA = 647.10 # Critical temperature of water in K 
T_C_EO = 469.15 # Critical temperature of ethylene oxide in K 

c_p_EG_params = np.array([0, 33.1585, -25.9580, 0, 0, 0]) # Parameters for the specific heat capacity of ethylene glycol as a function of temperature
c_p_WA_params = np.array([0.2399, 12.8647, -33.6392, 104.7686, -155.4709, 92.3726]) # Parameters for the specific heat capacity of water as a function of temperature



# Heat capacity of ethylene glycol dependent on temperature in J/kg-K
def c_p_EG(T):
    return R_EG * (c_p_EG_params[0] / (1- (T / T_C_EG)) + c_p_EG_params[1] + c_p_EG_params[2] * (1 - (T / T_C_EG)) + c_p_EG_params[3] * (1 - (T / T_C_EG))**2 + c_p_EG_params[4] * (1 - (T / T_C_EG))**3 + c_p_EG_params[5] * (1 - (T / T_C_EG))**4) * 1000

# Heat capacity of water dependent on temperature in J/kg-K
def c_p_WA(T):
    return R_WA * (c_p_WA_params[0] / (1- (T / T_C_WA)) + c_p_WA_params[1] + c_p_WA_params[2] * (1 - (T / T_C_WA)) + c_p_WA_params[3] * (1 - (T / T_C_WA))**2 + c_p_WA_params[4] * (1 - (T / T_C_WA))**3 + c_p_WA_params[5] * (1 - (T / T_C_WA))**4) * 1000

c_p_EO_values = np.array([1.88, 1.91, 1.95, 2.00, 2.06, 2.15, 2.27, 2.40]) * 1000 # Specific heat capacity values for ethylene oxide in J/kg-K at the corresponding temperature values in
T_EO_values = np.array([-40, -20, 0, 20, 40, 60, 80, 100]) + 273.15 # Temperature values for the specific heat capacity of ethylene oxide in K

# Interpolate the specific heat capacity of ethylene oxide as a function of temperature
c_p_EO_interp = interp1d(T_EO_values, c_p_EO_values, kind='linear', fill_value=(c_p_EO_values[0], c_p_EO_values[-1]), bounds_error=False) 

# Ideal mixture of specific heat capacities in J kg-1 K-1
def c_p_mix(T, x_arr):
    x_WA = x_arr[0]
    x_EG = x_arr[1]
    x_EO = x_arr[2]

    w_EG = w(x_arr, x_EG, M_EG)
    w_WA = w(x_arr, x_WA, M_WA)
    w_EO = w(x_arr, x_EO, M_EO)

    return w_EG * c_p_EG(T) + w_WA * c_p_WA(T) + w_EO * c_p_EO_interp(T)

rho_EG_params = np.array([1305.5931, -1374.2561, 1691.0501, -665.0358]) # Parameters for the density of ethylene glycol as a function of temperature
rho_WA_params = np.array([1094.0233, -1813.2295, 3863.9557, -2479.8130]) # Parameters for the density of water as a function of temperature
rho_EO_params = np.array([757.9994, -286.5638, 583.1649, -177.0206]) # Parameters for the density of ethylene oxide as a function of temperature (not used in this code)

rho_EG_c = 325 # Critical density of ethylene glycol in kg/m3
rho_WA_c = 322 # Critical density of water in kg/m3
rho_EO_c = 314 # Critical density of ethylene oxide in kg/m3


# Density of water dependent on temperature in kg/m3
def rho_WA(T):
    return rho_WA_c + rho_WA_params[0] * (1 - (T / T_C_WA)) ** 0.35 + rho_WA_params[1] * (1 - (T / T_C_WA))**(2/3) + rho_WA_params[2] * (1 - (T / T_C_WA)) + rho_WA_params[3] * (1 - (T / T_C_WA))**(4/3)

# Density of ethylene glycol dependent on temperature in kg/m3
def rho_EG(T):
    if type(T) is np.ndarray:
        T_inp = T.copy()
    else:
        T_inp = T


    if type(T) is np.ndarray:
        for i, T_val in enumerate(T_inp):
            if T_val > (273.15 + 200):
                T_inp[i] = 200 + 273.15

    else:
        if T > (273.15 + 200):
            T_inp = (273.15 + 200) 

    return rho_EG_c + rho_EG_params[0] * (1 - (T / T_C_EG)) ** 0.35 + rho_EG_params[1] * (1 - (T / T_C_EG))**(2/3) + rho_EG_params[2] * (1 - (T / T_C_EG)) + rho_EG_params[3] * (1 - (T / T_C_EG))**(4/3)

# Density of ethylene oxide dependent on temperature in kg/m3
def rho_EO(T):
    if type(T) is np.ndarray:
        T_inp = T.copy()
    else:
        T_inp = T


    if type(T) is np.ndarray:
        for i, T_val in enumerate(T_inp):
            if T_val > (273.15 + 150):
                T_inp[i] = 150 + 273.15

    else:
        if T > (273.15 + 150):
            T_inp = (273.15 + 150) 

    return rho_EO_c + rho_EO_params[0] * (1 - (T_inp / T_C_EO)) ** 0.35 + rho_EO_params[1] * (1 - (T_inp / T_C_EO))**(2/3) + rho_EO_params[2] * (1 - (T_inp / T_C_EO)) + rho_EO_params[3] * (1 - (T_inp / T_C_EO))**(4/3)


# Mass fraction of component x in the mixture
def w(x_arr, x, M):
    x_WA = x_arr[0]
    x_EG = x_arr[1]
    x_EO = x_arr[2]
    return x * M / (x_EG * M_EG + x_WA * M_WA + x_EO * M_EO)

# Density of the mixture dependent on temperature and composition in kg/m3
def rho_mix(T, x_arr):
    x_WA = x_arr[0]
    x_EG = x_arr[1]
    x_EO = x_arr[2]

    w_EG = w(x_arr, x_EG, M_EG)
    w_WA = w(x_arr, x_WA, M_WA)
    w_EO = w(x_arr, x_EO, M_EO)

    return 1 / (w_EG / rho_EG(T) + w_WA / rho_WA(T) + w_EO / rho_EO(T))

# Molar mass of the mixture in g/mol
def M_mix(x_arr):
    x_WA = x_arr[0]
    x_EG = x_arr[1]
    x_EO = x_arr[2]
    return x_WA * M_WA + x_EG * M_EG + x_EO * M_EO

lambda_EG_params = np.array([0.1125, 0.06626, -0.00088, -0.023, 0.01597]) # bis 150 °C
lambda_WA_params = np.array([-2.4149, 2.45165, -0.73121, 0.99492, -0.53730]) # bis 200 °C

def lambda_EG(T):
    if type(T) is np.ndarray:
        T_inp = T.copy()
    else:
        T_inp = T


    if type(T) is np.ndarray:
        for i, T_val in enumerate(T_inp):
            if T_val > (273.15 + 150):
                T_inp[i] = 150 + 273.15

    else:
        if T > (273.15 + 150):
            T_inp = (273.15 + 150) 

    return lambda_EG_params[0] + lambda_EG_params[1] / (10 ** 2) * T + lambda_EG_params[2] / (10 ** 4) * (T ** 2) + lambda_EG_params[3] / (10 ** 7) * (T ** 3) + lambda_EG_params[4] / (10 ** 10) * (T ** 4) 

def lambda_WA(T):
    return lambda_WA_params[0] + lambda_WA_params[1] / (10 ** 2) * T + lambda_WA_params[2] / (10 ** 4) * (T ** 2) + lambda_WA_params[3] / (10 ** 7) * (T ** 3) + lambda_WA_params[4] / (10 ** 10) * (T ** 4) 


lambda_EO_values = np.array([15.3, 15, 14.7, 14.3, 14.0, 13.7, 13.4, 12.5]) * 0.01 # Heat conductivity for ethylene oxide in W m-1 K-1
T_EG_values = np.array([20, 30, 40, 50, 60, 70, 80, 100]) + 273.15

# Interpolate the heat conductivity of ethylene oxide as a function of temperature
lambda_EO_interp = interp1d(T_EG_values, lambda_EO_values, kind='linear', fill_value=(lambda_EO_values[0], lambda_EO_values[-1]), bounds_error=False) 

# Thermal conductivity of the mixture dependent on temperature and composition in W m-1 K-1
def lambda_mix(T, x_arr):
    x_WA = x_arr[0]
    x_EG = x_arr[1]
    x_EO = x_arr[2]

    return x_EG * lambda_EG(T) + x_WA * lambda_WA(T) + x_EO * lambda_EO_interp(T)


eta_EG_params = np.array([-0.44356, 1.31885, 895.784, 79.362, 0.00020318]) # bis 150
eta_WA_params = np.array([0.45047, 1.39753, 613.181, 63.697, 0.00006896]) # bis 200

def eta_EG(T):
    if type(T) is np.ndarray:
        T_inp = T.copy()
    else:
        T_inp = T


    if type(T) is np.ndarray:
        for i, T_val in enumerate(T_inp):
            if T_val > (273.15 + 150):
                T_inp[i] = 150 + 273.15

    else:
        if T > (273.15 + 150):
            T_inp = (273.15 + 150) 

    if type(T_inp) is np.ndarray:
        frac = (eta_EG_params[2] - T_inp) / (T_inp - eta_EG_params[3]) 

        frac_root = np.array([])
        for i, frac_ in enumerate(frac):
            if frac_ < 0:
                frac_root = np.append(frac_root, -1 * (-frac_) ** (1 / 3))
        
            else:
                frac_root =  np.append(frac_root, frac_ ** (1/3))
    
    else: 
        frac = (eta_EG_params[2] - T_inp) / (T_inp - eta_EG_params[3]) 
        if frac < 0:
            frac_root = -1 * (-frac) ** (1 / 3)
        
        else:
            frac_root = frac ** (1/3)
    
    return eta_EG_params[4] * np.exp(eta_EG_params[0]* frac_root + eta_EG_params[1]* frac_root * frac)

def eta_WA(T):
    if type(T) is np.ndarray:
        T_inp = T.copy()
    else:
        T_inp = T


    if type(T) is np.ndarray:
        for i, T_val in enumerate(T_inp):
            if T_val > (273.15 + 200):
                T_inp[i] = 200 + 273.15

    else:
        if T > (273.15 + 200):
            T_inp = (273.15 + 200) 

    if type(T_inp) is np.ndarray:
        frac = (eta_WA_params[2] - T_inp) / (T_inp - eta_WA_params[3]) 

        frac_root = np.array([])
        for i, frac_ in enumerate(frac):
            if frac_ < 0:
                frac_root = np.append(frac_root, -1 * (-frac_) ** (1 / 3))
        
            else:
                frac_root =  np.append(frac_root, frac_ ** (1/3))
    
    else: 
        frac = (eta_WA_params[2] - T_inp) / (T_inp - eta_WA_params[3]) 
        if frac < 0:
            frac_root = -1 * (-frac) ** (1 / 3)
        
        else:
            frac_root = frac ** (1/3)
    
    return eta_WA_params[4] * np.exp(eta_WA_params[0]* frac_root + eta_WA_params[1]* frac_root * frac)

eta_EO_values = np.array([0.25, 0.23, 0.21, 0.19, 0.18, 0.16, 0.15, 0.14]) * 10 ** (-3)
eta_EO_interp = interp1d(T_EG_values, eta_EO_values, kind='linear', fill_value=(eta_EO_values[0], eta_EO_values[-1]), bounds_error=False) 

def eta_mix(T, x_arr):
    x_WA = x_arr[0]
    x_EG = x_arr[1]
    x_EO = x_arr[2]

    return x_EG * eta_EG(T) + x_WA * eta_WA(T) + x_EO * eta_EO_interp(T)
    


def h_by_petukov(T, x, d_R):
    Re = rho_mix(T, x) * u * d_R / eta_mix(T, x)
    Pr = eta_mix(T, x) * c_p_mix(T, x) / lambda_mix(T, x)
    xi = np.power(1.8 * np.log10(Re) - 1.5, -2)
    Nu = np.divide(((xi/8) * (Re - 1000) * Pr),(1 + 12.7 * np.power(xi/8, 0.5) * (np.power(Pr, 2/3) - 1))) * (1 + np.power(d_R / L_R, 2/3))
    h = Nu * lambda_mix(T, x) / d_R
    return h



# Calculated molar flow in mol/s
def n_dot(u, rho, M):
    return u * rho * A_ref / (M * 0.001)

# Calculated liquid velocity in m/s
def u(n_dot, rho, M):
    return n_dot * M / (rho * A_ref * 1000)

# Molar fraction calculated from concentrations
def x_i(c_arr, c_i):
    return c_i / np.sum(c_arr)

# Function for calculation molar flow of water in mol/s  via root finding
def search_WA_inlet(n_dot_in_WA, n_dot_in_EO, T_0, u_ref):
    x_WA = n_dot_in_WA / (n_dot_in_WA + n_dot_in_EO)
    x_EO = n_dot_in_EO / (n_dot_in_WA + n_dot_in_EO)

    x_arr = np.array([x_WA[0], 0, x_EO[0]])

    u_ = u(n_dot_in_WA + n_dot_in_EO, rho_mix(T_0, x_arr), M_mix(x_arr))

    return u_ - u_ref


#%%
# Sensitivity 4: Tube Diameter — h computed from Petukhov correlation

# Initial water concentration from density balance
# rho_mix ≈ 1000 kg/m³ → mass of WA = rho - c_EO*M_EO/1000
c_WA_0 = (rho - c_10 * M_EO / 1000) / (M_WA / 1000)  # mol/m³  ≈ 49970

def PFTR_petukhov(L, f, d):
    """
    PFTR with Petukhov h, tracking EO, EG, H2O, T.
    f[0] = c_EO  (mol/m³)
    f[1] = c_EG  (mol/m³)
    f[2] = c_WA  (mol/m³)
    f[3] = T     (K)
    d    = tube diameter (m)
    """
    c_EO, c_EG, c_WA, T = f[0], f[1], f[2], f[3]

    # Mole fractions for mixture properties  [x_WA, x_EG, x_EO]
    c_tot = c_EO + c_EG + c_WA
    x_arr = np.array([c_WA / c_tot, c_EG / c_tot, c_EO / c_tot])

    # Position-dependent h and a_wall for this diameter
    h  = h_by_petukov(T, x_arr, d)
    a  = 4 / d                                          # specific wall area m²/m³

    # Composition-dependent volumetric heat capacity
    C_mix = rho_mix(T, x_arr) * c_p_mix(T, x_arr)     # J/(m³ K)

    r = kinetics(T, c_EO)

    dc_EOdL = -r / u
    dc_EGdL =  r / u
    dc_WAdL = -r / u   # water consumed 1:1 with EO

    dTdL = (-H_R * r / C_mix - h * a * (T - T_wall) / C_mix) / u

    return np.array([dc_EOdL, dc_EGdL, dc_WAdL, dTdL])


d_cases = {
    '$d_R = 0.01$ m':        0.01,
    '$d_R = 0.03$ m (base)': 0.03,
    '$d_R = 0.05$ m':        0.05,
    '$d_R = 0.10$ m':        0.10,
    '$d_R = 0.20$ m':        0.20,
}

f_init_4 = np.array([c_10, c_20, c_WA_0, T_0])
colors   = ['tab:purple', 'tab:blue', 'tab:green', 'tab:orange', 'tab:red']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
ax1.set(xlabel='$L$ / m', ylabel='$c_{\\mathrm{EO}}$ / mol L$^{-1}$')
ax2.set(xlabel='$L$ / m', ylabel='$T$ / K')
ax3.set(xlabel='$L$ / m', ylabel='$h$ / W m$^{-2}$ K$^{-1}$')

for (label, d), color in zip(d_cases.items(), colors):
    sol = integ.solve_ivp(
        PFTR_petukhov, Lspan, f_init_4,
        args=(d,), method='BDF', t_eval=leval
    )

    # Reconstruct h along the reactor for the third panel
    c_tot = sol.y[0] + sol.y[1] + sol.y[2]
    x_WA  = sol.y[2] / c_tot
    x_EG  = sol.y[1] / c_tot
    x_EO  = sol.y[0] / c_tot
    T_sol = sol.y[3]
    h_sol = np.array([
        h_by_petukov(T_sol[i], np.array([x_WA[i], x_EG[i], x_EO[i]]), d)
        for i in range(len(sol.t))
    ])

    ax1.plot(sol.t, sol.y[0] / 1000,  color=color, label=label)
    ax2.plot(sol.t, T_sol,             color=color, label=label)
    ax3.plot(sol.t, h_sol,             color=color, label=label)

for ax in (ax1, ax2, ax3):
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

#%%
# Sensitivity 4b: Tube Diameter — constant volumetric flow rate V

# Base volumetric flow from the base case (d_R=0.03, u=1.0 m/s)
V_base = u * np.pi * (d_R / 2)**2   # m³/s  ≈ 7.07e-4

def PFTR_petukhov_Q(L, f, d, u_d):
    """
    Same as PFTR_petukhov but with variable fluid velocity u_d,
    used for constant-Q diameter sensitivity.
    """
    c_EO, c_EG, c_WA, T = f[0], f[1], f[2], f[3]

    c_tot = c_EO + c_EG + c_WA
    x_arr = np.array([c_WA / c_tot, c_EG / c_tot, c_EO / c_tot])

    h  = h_by_petukov(T, x_arr, d)
    a  = 4 / d
    C_mix = rho_mix(T, x_arr) * c_p_mix(T, x_arr)

    r = kinetics(T, c_EO)

    dc_EOdL = -r / u_d
    dc_EGdL =  r / u_d
    dc_WAdL = -r / u_d
    dTdL    = (-H_R * r / C_mix - h * a * (T - T_wall) / C_mix) / u_d

    return np.array([dc_EOdL, dc_EGdL, dc_WAdL, dTdL])


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
ax1.set(xlabel='$L$ / m', ylabel='$c_{\\mathrm{EO}}$ / mol L$^{-1}$')
ax2.set(xlabel='$L$ / m', ylabel='$T$ / K')
ax3.set(xlabel='$L$ / m', ylabel='$h$ / W m$^{-2}$ K$^{-1}$')

for (label, d), color in zip(d_cases.items(), colors):
    u_d = V_base / (np.pi * (d / 2)**2)   # velocity for this diameter at constant Q

    sol = integ.solve_ivp(
        PFTR_petukhov_Q, Lspan, f_init_4,
        args=(d, u_d), method='BDF', t_eval=leval
    )

    c_tot = sol.y[0] + sol.y[1] + sol.y[2]
    x_WA  = sol.y[2] / c_tot
    x_EG  = sol.y[1] / c_tot
    x_EO  = sol.y[0] / c_tot
    T_sol = sol.y[3]
    h_sol = np.array([
        h_by_petukov(T_sol[i], np.array([x_WA[i], x_EG[i], x_EO[i]]), d)
        for i in range(len(sol.t))
    ])

    # Add velocity to the label so the difference is immediately visible
    label_V = label + f'\n$u = {u_d:.2f}$ m s$^{{-1}}$'
    ax1.plot(sol.t, sol.y[0] / 1000, color=color, label=label_V)
    ax2.plot(sol.t, T_sol,            color=color, label=label_V)
    ax3.plot(sol.t, h_sol,            color=color, label=label_V)

for ax in (ax1, ax2, ax3):
    ax.legend(fontsize=7)
fig.suptitle(f'Constant volumetric flow $V = {V_base*1000:.3f}$ L s$^{{-1}}$', y=1.01)
fig.tight_layout()
plt.show()