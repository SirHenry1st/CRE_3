#%%
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


#%%

R_WA = 8.314 / 18 # Specific heat capacity of water in J/mol-K
R_EG = 8.314 / 62.07 # Specific heat capacity of ethylene glycol in J/mol-K

T_C_EG = 724.05 # Critical temperature of ethylene glycol
T_C_WA = 647.10 # Critical temperature of water in K 

c_p_EG_params = np.array([0, 33.1585, -25.9580, 0, 0, 0]) # Parameters for the specific heat capacity of ethylene glycol as a function of temperature
c_p_WA_params = np.array([0.2399, 12.8647, -33.6392, 104.7686, -155.4709, 92.3726]) # Parameters for the specific heat capacity of water as a function of temperature

# Heat capacity of ethylene glycol dependent on temperature
def c_p_EG(T, c_p_EG_params):
    return R_EG * (c_p_EG_params[0] / (1- (T / T_C_EG)) + c_p_EG_params[1] + c_p_EG_params[2] * (1 - (T / T_C_EG)) + c_p_EG_params[3] * (1 - (T / T_C_EG))**2 + c_p_EG_params[4] * (1 - (T / T_C_EG))**3 + c_p_EG_params[5] * (1 - (T / T_C_EG))**4) * 1000

# Heat capacity of water dependent on temperature
def c_p_WA(T, c_p_WA_params):
    return R_WA * (c_p_WA_params[0] / (1- (T / T_C_WA)) + c_p_WA_params[1] + c_p_WA_params[2] * (1 - (T / T_C_WA)) + c_p_WA_params[3] * (1 - (T / T_C_WA))**2 + c_p_WA_params[4] * (1 - (T / T_C_WA))**3 + c_p_WA_params[5] * (1 - (T / T_C_WA))**4) * 1000

# Ideal mixture of specific heat capacities in J kg-1 K-1
def c_p_mix(T, c_p_EG_params, c_p_WA_params, x_arr):
    x_EG = x_arr[0]
    x_WA = x_arr[1]
    return x_EG * c_p_EG(T, c_p_EG_params) + x_WA * c_p_WA(T, c_p_WA_params)

rho_EG_params = np.array([1305.5931, -1374.2561, 1691.0501, -665.0358]) # Parameters for the density of ethylene glycol as a function of temperature
rho_WA_params = np.array([1094.0233, -1813.2295, 3863.9557, -2479.8130]) # Parameters for the density of water as a function of temperature

rho_EG_c = 325 # Critical density of ethylene glycol in kg/m^3
rho_WA_c = 322 # Critical density of water in kg/m^3

# Density of water dependent on temperature
def rho_WA(T, rho_WA_params):
    return rho_WA_c + rho_WA_params[0] * (1 - (T / T_C_WA)) ** 0.35 + rho_WA_params[1] * (1 - (T / T_C_WA))**(2/3) + rho_WA_params[2] * (1 - (T / T_C_WA)) + rho_WA_params[3] * (1 - (T / T_C_WA))**(4/3)

# Density of ethylene glycol dependent on temperature
def rho_EG(T, rho_EG_params):
    return rho_EG_c + rho_EG_params[0] * (1 - (T / T_C_EG)) ** 0.35 + rho_EG_params[1] * (1 - (T / T_C_EG))**(2/3) + rho_EG_params[2] * (1 - (T / T_C_EG)) + rho_EG_params[3] * (1 - (T / T_C_EG))**(4/3)

print(rho_EG(273.15 + 20, rho_EG_params))
print(rho_WA(273.15 + 20, rho_WA_params))


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 12))
T_range = np.arange(293, 500, 1) 
ax[0].plot(T_range, c_p_EG(T_range, c_p_EG_params), label='Ethylene Glycol')
ax[0].plot(T_range, c_p_WA(T_range, c_p_WA_params), label='Water')
ax[0].set_xlabel('Temperature (K)')
ax[0].set_ylabel('Specific Heat Capacity (J/kg-K)')
ax[0].set_title('Specific Heat Capacity vs. Temperature')
ax[0].legend()

ax[1].plot(T_range, rho_EG(T_range, rho_EG_params), label='Ethylene Glycol')
ax[1].plot(T_range, rho_WA(T_range, rho_WA_params), label='Water')
ax[1].set_xlabel('Temperature (K)')
ax[1].set_ylabel('Density (kg/m^3)')
ax[1].set_title('Density vs. Temperature')
ax[1].legend()
plt.show()

# Function that adds an inlet flow rate of EO to the system at the location of the jet inlet
def multi_jet_inlet(L, x_arr, n_dot_in_EO):
    if L == x_arr.any():
        return n_dot_in_EO
    else:
        return 0
    

# %%
