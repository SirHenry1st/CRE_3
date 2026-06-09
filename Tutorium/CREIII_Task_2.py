#%% 
# Import
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import ICIW_Plots.colors as iciw_colors
from CoolProp.CoolProp import PropsSI
from numpy.linalg import matrix_rank

#%%
#Matrix

# define matrix of stoichiometric coefficients
N_stoich = np.array([[-1,  0,  -1],  # CO
                     [ 0, -1,   1],  # CO2
                     [ 1,  1,   0],  # CH4
                     [-3, -4,   1],  # H2
                     [ 1,  2,  -1]]) # H2O
# N_stoich.T is the transposed matrix

# determination of the rank of the matrix of stoichiometric coefficients
N_stoich_rank = matrix_rank(N_stoich)
print(r'Rank:',N_stoich_rank) # display key result
# split of the matrix of stoichiometric coefficients
N_stoich_11 = N_stoich[0:N_stoich_rank, 0:N_stoich_rank] # square sub-matrix of size rank
N_stoich_21 = N_stoich[N_stoich_rank:N_stoich.shape[0], 0:N_stoich_rank]
N_stoich_12 = N_stoich[0:N_stoich_rank, N_stoich_rank:N_stoich.shape[1]]
N_stoich_22 = N_stoich[N_stoich_rank:N_stoich.shape[0], N_stoich_rank:N_stoich.shape[1]]



#%%

# define function with arguments temperature and name of component
def prop_thermo(T, comp):
    results = np.empty(3) # generate empty vector of required size
    t=T/1000 # define the dimensionless temperature required
    # define the Shomate equations
    fun_cp_NIST = np.array([t**0, t, t**2, t**3, t**-2, 0, 0, 0])
    fun_H_NIST = np.array([t, t**2/2, t**3/3, t**4/4, -t**-1, 1, 0, -1])
    fun_S_NIST = np.array([np.log(t), t, t**2/2, t**3/3, -t**-2/2, 0, 1, 0])
    # assign parameters of the Shomate equation to components
    if comp == 0 or comp == 'H2': #
        PhysProp_param_NIST = np.array([33.07, -11.36, 11.43, -2.773, -.1586, -9.981, 172.7, 0])
    elif comp == 1 or comp == 'CO': #
        PhysProp_param_NIST = np.array([25.568, 6.096, 4.055, -2.671, 0.131, -118.009, 227.367, -110.527])
    elif comp == 2 or comp == 'CO2': #
        PhysProp_param_NIST = np.array([24.997, 55.187, -33.691, 7.948, -0.137, -403.608, 228.24, -393.522])
    elif comp == 3 or comp == 'CH4': #
        PhysProp_param_NIST = np.array([-0.703, 108.477, -42.522, 5.863, 0.679, -76.844, 158.716, -74.873])
    elif comp == 4 or comp == 'H2O': #
        PhysProp_param_NIST = np.array([30.092, 6.833, 6.793, -2.534, 0.082, -250.881, 223.397, -241.826])
    else:
        PhysProp_param_NIST = np.zeros(8) # default case for non-existing component
    # multiply set of parameters to Shomate equation
    results[0] = fun_cp_NIST.dot(PhysProp_param_NIST) # heat capacity
    results[1] = (PhysProp_param_NIST[-1] + fun_H_NIST.dot(PhysProp_param_NIST))*1000 # enthalpy of formation
    results[2] = fun_S_NIST.dot(PhysProp_param_NIST) # entropy
    return results
# call of the function for T = 25°C and CO to compare with tabulated data
print('vector of heat capacity, enthalpy and entropy for carbonmonoxide:',prop_thermo(25+273, 'CO')) # display of key result for check


#%%
# CoolProp

def prop_thermo_CP(T, comp, P=101325):

    results = np.empty(3)

    # mapping between short names and CoolProp fluids
    fluid_map = {
        'H2':  'Hydrogen',
        'CO':  'CarbonMonoxide',
        'CO2': 'CarbonDioxide',
        'CH4': 'Methane',
        'H2O': 'Water'
    }

    # allow integer indexing like your original code
    if comp == 0:
        comp = 'H2'
    elif comp == 1:
        comp = 'CO'
    elif comp == 2:
        comp = 'CO2'
    elif comp == 3:
        comp = 'CH4'
    elif comp == 4:
        comp = 'H2O'

    # unknown component handling
    if comp not in fluid_map:
        return np.zeros(3)

    fluid = fluid_map[comp]

    results[0] = PropsSI(
        'CP0MOLAR',
        'T', T,
        'P', P,
        fluid
    )

    results[1] = PropsSI(
        'HMOLAR',
        'T', T,
        'P', P,
        fluid
    )

    results[2] = PropsSI(
        'SMOLAR',
        'T', T,
        'P', P,
        fluid
    )

    return results


T = 25 + 273.15

print(
    'vector of heat capacity, enthalpy and entropy for carbon monoxide:',
    prop_thermo_CP(T, 'CO')
)

#%%
# Reference state correction

Tref = 298.15
Pref = 101325

fluid_map = {
    'H2':  'Hydrogen',
    'CO':  'CarbonMonoxide',
    'CO2': 'CarbonDioxide',
    'CH4': 'Methane',
    'H2O': 'Water'
}


# store offsets
offsets = {}

for comp in fluid_map:

    # NIST reference values
    nist = prop_thermo(Tref, comp)

    H_nist = nist[1]
    S_nist = nist[2]

    # CoolProp reference values
    H_cp = PropsSI(
        'HMOLAR',
        'T', Tref,
        'P', Pref,
        fluid_map[comp]
    )

    S_cp = PropsSI(
        'SMOLAR',
        'T', Tref,
        'P', Pref,
        fluid_map[comp]
    )

    offsets[comp] = {
        'dH': H_nist - H_cp,
        'dS': S_nist - S_cp
    }


# -----------------------------------
# corrected CoolProp properties
# -----------------------------------

def prop_thermo_CP(T, comp, P=101325):

    results = np.empty(3)

    fluid = fluid_map[comp]

    cp = PropsSI(
        'CPMOLAR',
        'T', T,
        'P', P,
        fluid
    )

    h = PropsSI(
        'HMOLAR',
        'T', T,
        'P', P,
        fluid
    )

    s = PropsSI(
        'SMOLAR',
        'T', T,
        'P', P,
        fluid
    )

    # apply reference-state correction
    h += offsets[comp]['dH']
    s += offsets[comp]['dS']

    results[0] = cp
    results[1] = h
    results[2] = s

    return results


print(
    'vector of heat capacity, enthalpy and entropy for carbon monoxide:',
    prop_thermo_CP(T, 'CO')
)

#%%

def rxn_data_methanation(T, P=101325):

    res = np.empty(5)

    # component thermodynamic data
    data_CO2 = prop_thermo_CP(T, 'CO2', P)
    data_H2  = prop_thermo_CP(T, 'H2',  P)
    data_CH4 = prop_thermo_CP(T, 'CH4', P)
    data_H2O = prop_thermo_CP(T, 'H2O', P)

    # -----------------------------------
    # reaction thermodynamics
    # -----------------------------------

    # ΔCp
    res[0] = (
        data_CH4[0]
        + data_H2O[0]
        - data_CO2[0]
        - 4*data_H2[0]
    )

    # ΔH
    res[1] = (
        data_CH4[1]
        + data_H2O[1]
        - data_CO2[1]
        - 4*data_H2[1]
    )

    # ΔS
    res[2] = (
        data_CH4[2]
        + data_H2O[2]
        - data_CO2[2]
        - 4*data_H2[2]
    )

    # ΔG
    res[3] = res[1] - T*res[2]

    # equilibrium constant
    res[4] = np.exp(
        -res[3] / (8.314462618 * T)
    )

    return res


# -------------------------------------------------
# example at 25°C
# -------------------------------------------------

T = 25 + 273.15

print(
    'vector with ΔCp, ΔH, ΔS, ΔG and K_eq:',
    rxn_data_methanation(T)
)

#%%
# define vectors for typical ranges of T and p
T = np.linspace(400+273, 500+273, 100) # temperature in K
p = np.array([100, 200, 300]) # total pressure in bar

K_x = np.empty([p.shape[0],T.shape[0]]) # generate empty vector of required size
for TT in range(T.shape[0]): # vary the temperature within the given range
    # at each single temperature
    rxn_data_get = rxn_data_methanation(T[TT]) # call the function for thermodynamic data of the reaction 
    K_eq = rxn_data_get[-1] # the last element in the results vector provides the thermodynamic equilibrium constant
    K_x[:,TT] = K_eq*np.power(p,2) # calculate K_x at each temperature for all pressures

# generate the plot
plt.figure(figsize=(5, 5))
plt.grid()
plt.xlabel(r"$T \, / \,°C$")
plt.ylabel("$K_x\,/\,1$")
plt.plot(T[:]-273, K_x[0,:], 'r-', label="$100$ bar")
plt.plot(T[:]-273, K_x[1,:], 'g-', label="$200$ bar")
plt.plot(T[:]-273, K_x[2,:], 'b-', label="$300$ bar")
plt.legend(loc='best')
plt.show()