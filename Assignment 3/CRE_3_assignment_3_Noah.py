#%%
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


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
def c_p_EG(T, c_p_EG_params):
    return R_EG * (c_p_EG_params[0] / (1- (T / T_C_EG)) + c_p_EG_params[1] + c_p_EG_params[2] * (1 - (T / T_C_EG)) + c_p_EG_params[3] * (1 - (T / T_C_EG))**2 + c_p_EG_params[4] * (1 - (T / T_C_EG))**3 + c_p_EG_params[5] * (1 - (T / T_C_EG))**4) * 1000

# Heat capacity of water dependent on temperature in J/kg-K
def c_p_WA(T, c_p_WA_params):
    return R_WA * (c_p_WA_params[0] / (1- (T / T_C_WA)) + c_p_WA_params[1] + c_p_WA_params[2] * (1 - (T / T_C_WA)) + c_p_WA_params[3] * (1 - (T / T_C_WA))**2 + c_p_WA_params[4] * (1 - (T / T_C_WA))**3 + c_p_WA_params[5] * (1 - (T / T_C_WA))**4) * 1000

c_p_EO_values = np.array([1.88, 1.91, 1.95, 2.00, 2.06, 2.15, 2.27, 2.40]) * 1000 # Specific heat capacity values for ethylene oxide in J/kg-K at the corresponding temperature values in
T_EO_values = np.array([-40, -20, 0, 20, 40, 60, 80, 100]) + 273.15 # Temperature values for the specific heat capacity of ethylene oxide in K

# Interpolate the specific heat capacity of ethylene oxide as a function of temperature
c_p_EO_interp = interp1d(T_EO_values, c_p_EO_values, kind='linear', fill_value=(c_p_EO_values[0], c_p_EO_values[-1]), bounds_error=False) 


# Ideal mixture of specific heat capacities in J kg-1 K-1
def c_p_mix(T, c_p_EG_params, c_p_WA_params, x_arr):
    x_EG = x_arr[0]
    x_WA = x_arr[1]
    x_EO = x_arr[2]

    w_EG = w(x_arr, x_EG, M_EG)
    w_WA = w(x_arr, x_WA, M_WA)
    w_EO = w(x_arr, x_EO, M_EO)

    return w_EG * c_p_EG(T, c_p_EG_params) + w_WA * c_p_WA(T, c_p_WA_params) + w_EO * c_p_EO_interp(T)

rho_EG_params = np.array([1305.5931, -1374.2561, 1691.0501, -665.0358]) # Parameters for the density of ethylene glycol as a function of temperature
rho_WA_params = np.array([1094.0233, -1813.2295, 3863.9557, -2479.8130]) # Parameters for the density of water as a function of temperature
rho_EO_params = np.array([757.9994, -286.5638, 583.1649, -177.0206]) # Parameters for the density of ethylene oxide as a function of temperature (not used in this code)

rho_EG_c = 325 # Critical density of ethylene glycol in kg/m^3
rho_WA_c = 322 # Critical density of water in kg/m^3
rho_EO_c = 314 # Critical density of ethylene oxide in kg/m^3

# Density of water dependent on temperature
def rho_WA(T, rho_WA_params):
    return rho_WA_c + rho_WA_params[0] * (1 - (T / T_C_WA)) ** 0.35 + rho_WA_params[1] * (1 - (T / T_C_WA))**(2/3) + rho_WA_params[2] * (1 - (T / T_C_WA)) + rho_WA_params[3] * (1 - (T / T_C_WA))**(4/3)

# Density of ethylene glycol dependent on temperature
def rho_EG(T, rho_EG_params):
    return rho_EG_c + rho_EG_params[0] * (1 - (T / T_C_EG)) ** 0.35 + rho_EG_params[1] * (1 - (T / T_C_EG))**(2/3) + rho_EG_params[2] * (1 - (T / T_C_EG)) + rho_EG_params[3] * (1 - (T / T_C_EG))**(4/3)

def rho_EO(T, rho_EO_params):
    return rho_EO_c + rho_EO_params[0] * (1 - (T / T_C_EO)) ** 0.35 + rho_EO_params[1] * (1 - (T / T_C_EO))**(2/3) + rho_EO_params[2] * (1 - (T / T_C_EO)) + rho_EO_params[3] * (1 - (T / T_C_EO))**(4/3)

print(rho_EG(273.15 + 20, rho_EG_params))
print(rho_WA(273.15 + 20, rho_WA_params))
print(rho_EO(273.15 + 20, rho_EO_params))

# Mass fraction of component x in the mixture
def w(x_arr, x, M):
    x_WA = x_arr[0]
    x_EG = x_arr[1]
    x_EO = x_arr[2]
    return x * M / (x_EG * M_EG + x_WA * M_WA + x_EO * M_EO)

# Density of the mixture dependent on temperature and composition
def rho_mix(T, rho_EG_params, rho_WA_params, rho_EO_params, x_arr):
    x_WA = x_arr[0]
    x_EG = x_arr[1]
    x_EO = x_arr[2]

    w_EG = w(x_arr, x_EG, M_EG)
    w_WA = w(x_arr, x_WA, M_WA)
    w_EO = w(x_arr, x_EO, M_EO)

    return 1 / (w_EG / rho_EG(T, rho_EG_params) + w_WA / rho_WA(T, rho_WA_params) + w_EO / rho_EO(T, rho_EO_params))

def x_i(c_arr, c_i):
    return c_i / np.sum(c_arr)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(30, 12))
T_range = np.arange(20, 150, 1) + 273.15
ax[0, 0].plot(T_range - 273.15, c_p_EG(T_range, c_p_EG_params), label='Ethylene Glycol')
ax[0, 0].plot(T_range - 273.15, c_p_WA(T_range, c_p_WA_params), label='Water')
ax[0, 0].plot(T_range - 273.15, c_p_EO_interp(T_range), label='Ethylene Oxide')
ax[0, 0].set_xlabel('Temperature (°C)')
ax[0, 0].set_ylabel('Specific Heat Capacity (J/kg-K)')
ax[0, 0].set_title('Specific Heat Capacity vs. Temperature')
ax[0, 0].legend()

ax[0, 1].plot(T_range - 273.15, rho_EG(T_range, rho_EG_params), label='Ethylene Glycol')
ax[0, 1].plot(T_range - 273.15, rho_WA(T_range, rho_WA_params), label='Water')
ax[0, 1].plot(T_range - 273.15, rho_EO(T_range, rho_EO_params), label='Ethylene Oxide')
ax[0, 1].set_xlabel('Temperature (°C)')
ax[0, 1].set_ylabel('Density (kg/m^3)')
ax[0, 1].set_title('Density vs. Temperature')
ax[0, 1].legend()

ax[1, 0].plot(T_range - 273.15, c_p_mix(T_range, c_p_EG_params, c_p_WA_params, np.array([0.5, 0.5, 0])), label='50 mol% EG, 50 mol% WA')
ax[1, 0].plot(T_range - 273.15, c_p_mix(T_range, c_p_EG_params, c_p_WA_params, np.array([0.5, 0, 0.5])), label='50 mol% EG, 50 mol% EO')
ax[1, 0].plot(T_range - 273.15, c_p_mix(T_range, c_p_EG_params, c_p_WA_params, np.array([0, 0.5, 0.5])), label='50 mol% EO, 50 mol% WA')
ax[1, 0].set_xlabel('Temperature (°C)')
ax[1, 0].set_ylabel('Specific Heat Capacity (J/kg-K)')
ax[1, 0].set_title('Specific Heat Capacity of Mixture vs. Temperature')
ax[1, 0].legend() 

ax[1, 1].plot(T_range - 273.15, rho_mix(T_range, rho_EG_params, rho_WA_params, rho_EO_params, np.array([0.5, 0.5, 0])), label='50 mol% EG, 50 mol% WA')
ax[1, 1].plot(T_range - 273.15, rho_mix(T_range, rho_EG_params, rho_WA_params, rho_EO_params, np.array([0.5, 0, 0.5])), label='50 mol% EG, 50 mol% EO')
ax[1, 1].plot(T_range - 273.15, rho_mix(T_range, rho_EG_params, rho_WA_params, rho_EO_params, np.array([0, 0.5, 0.5])), label='50 mol% EO, 50 mol% WA')
ax[1, 1].set_xlabel('Temperature (°C)')
ax[1, 1].set_ylabel('Density (kg/m^3)')
ax[1, 1].set_title('Density of Mixture vs. Temperature')
ax[1, 1].legend()
plt.show()
#%%

# Locations of the injections based on the length of the system and the distance between injections, initial injection not included
def injection_locations(L, N):
    delta_L = L / (N + 1)
    loc = np.array([])
    for i in range(1, N + 1):
        print(i)
        loc = np.append(loc, i * delta_L)
    return loc

# Function that adds an inlet flow rate of EO to the system at the location of the jet inle
def multi_jet_inlet(x, L, N, n_dot_in_EO):
    n_dot_in_EO_injection = n_dot_in_EO / (N + 1) # Flow rate of EO at each injection point
    x_arr = injection_locations(L, N)
    if x == x_arr.any():
        return n_dot_in_EO_injection
    else:
        return 0
    
def initial_injection(n_dot_in_EO , N):
    return n_dot_in_EO / (N + 1) # Initial flow rate of EO at the first injection point
    

# %%
