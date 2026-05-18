#%%
# Imports of packages
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import ICIW_Plots.colors as iciw_colors
from CoolProp.CoolProp import PropsSI
import numpy as np
from numpy.linalg import matrix_rank

# ---------------------------------------------------------
# 1. Define components and reactions
# ---------------------------------------------------------

components = np.array(["CO2", "H2", "CH3OH", "H2O", "CO", "DME"])

reactions = np.array([
    "R1: CO2 + 3H2 -> CH3OH + H2O",
    "R2: CO + 2H2 -> CH3OH",
    "R3: CO2 + H2 -> CO + H2O",
    "R4: 2CH3OH -> DME + H2O"
])

# ---------------------------------------------------------
# 2. Stoichiometric matrix
# Rows = components
# Columns = reactions
# ---------------------------------------------------------

N_stoich = np.array([
    [-1,  0, -1,  0],   # CO2
    [-3, -2, -1,  0],   # H2
    [ 1,  1,  0, -2],   # CH3OH
    [ 1,  0,  1,  1],   # H2O
    [ 0, -1,  1,  0],   # CO
    [ 0,  0,  0,  1]    # DME
], dtype=float)

print("Stoichiometric matrix N:")
print(N_stoich)

# ---------------------------------------------------------
# 3. Rank of the stoichiometric matrix
# ---------------------------------------------------------

rank_N = matrix_rank(N_stoich)

print("\nRank of full stoichiometric matrix:", rank_N)

# ---------------------------------------------------------
# 4. Check dependency of R3
# R3 should be equal to R1 - R2
# ---------------------------------------------------------

print("\nCheck stoichiometric dependency:")
print("R3 column:")
print(N_stoich[:, 2])

print("\nR1 - R2 column:")
print(N_stoich[:, 0] - N_stoich[:, 1])

print("\nIs R3 = R1 - R2?")
print(np.allclose(N_stoich[:, 2], N_stoich[:, 0] - N_stoich[:, 1]))

# ---------------------------------------------------------
# 5. Choose key reactions and key components
# Key reactions: R1, R2, R4
# Key components: CO2, CO, DME
# ---------------------------------------------------------

key_reaction_idx = [0, 1, 3]     # R1, R2, R4
key_component_idx = [0, 4, 5]    # CO2, CO, DME

N11 = N_stoich[np.ix_(key_component_idx, key_reaction_idx)]

print("\nChosen key reactions:")
print(reactions[key_reaction_idx])

print("\nChosen key components:")
print(components[key_component_idx])

print("\nReduced key matrix N11:")
print(N11)

print("\nRank of N11:", matrix_rank(N11))

if matrix_rank(N11) == rank_N:
    print("\nThe chosen key components and key reactions are valid.")
else:
    print("\nThe chosen set is not valid. Another set must be selected.")

# ---------------------------------------------------------
# 6. Verification using atom balance
# Atom matrix order: C, H, O
# Component order: CO2, H2, CH3OH, H2O, CO, DME
# ---------------------------------------------------------

atom_matrix = np.array([
    [1, 0, 1, 0, 1, 2],   # Carbon atoms
    [0, 2, 4, 2, 0, 6],   # Hydrogen atoms
    [2, 0, 1, 1, 1, 1]    # Oxygen atoms
], dtype=float)

atom_balance_check = atom_matrix @ N_stoich

print("\nAtom balance check:")
print(atom_balance_check)

if np.allclose(atom_balance_check, 0):
    print("Atom balance is satisfied for all reactions.")
else:
    print("Atom balance error detected.")

import numpy as np
from numpy.linalg import matrix_rank

# ---------------------------------------------------------
# 1. Define components and reactions
# ---------------------------------------------------------

components = np.array(["CO2", "H2", "CH3OH", "H2O", "CO", "DME"])

reactions = np.array([
    "R1: CO2 + 3H2 -> CH3OH + H2O",
    "R2: CO + 2H2 -> CH3OH",
    "R3: CO2 + H2 -> CO + H2O",
    "R4: 2CH3OH -> DME + H2O"
])

# ---------------------------------------------------------
# 2. Stoichiometric matrix
# Rows = components
# Columns = reactions
# ---------------------------------------------------------

N_stoich = np.array([
    [-1,  0, -1,  0],   # CO2
    [-3, -2, -1,  0],   # H2
    [ 1,  1,  0, -2],   # CH3OH
    [ 1,  0,  1,  1],   # H2O
    [ 0, -1,  1,  0],   # CO
    [ 0,  0,  0,  1]    # DME
], dtype=float)

print("Stoichiometric matrix N:")
print(N_stoich)

# ---------------------------------------------------------
# 3. Rank of the stoichiometric matrix
# ---------------------------------------------------------

rank_N = matrix_rank(N_stoich)

print("\nRank of full stoichiometric matrix:", rank_N)

# ---------------------------------------------------------
# 4. Check dependency of R3
# R3 should be equal to R1 - R2
# ---------------------------------------------------------

print("\nCheck stoichiometric dependency:")
print("R3 column:")
print(N_stoich[:, 2])

print("\nR1 - R2 column:")
print(N_stoich[:, 0] - N_stoich[:, 1])

print("\nIs R3 = R1 - R2?")
print(np.allclose(N_stoich[:, 2], N_stoich[:, 0] - N_stoich[:, 1]))

# ---------------------------------------------------------
# 5. Choose key reactions and key components
# Key reactions: R1, R2, R4
# Key components: CO2, CO, DME
# ---------------------------------------------------------

key_reaction_idx = [0, 1, 3]     # R1, R2, R4
key_component_idx = [0, 4, 5]    # CO2, CO, DME

N11 = N_stoich[np.ix_(key_component_idx, key_reaction_idx)]

print("\nChosen key reactions:")
print(reactions[key_reaction_idx])

print("\nChosen key components:")
print(components[key_component_idx])

print("\nReduced key matrix N11:")
print(N11)

print("\nRank of N11:", matrix_rank(N11))

if matrix_rank(N11) == rank_N:
    print("\nThe chosen key components and key reactions are valid.")
else:
    print("\nThe chosen set is not valid. Another set must be selected.")

# ---------------------------------------------------------
# 6. Verification using atom balance
# Atom matrix order: C, H, O
# Component order: CO2, H2, CH3OH, H2O, CO, DME
# ---------------------------------------------------------

atom_matrix = np.array([
    [1, 0, 1, 0, 1, 2],   # Carbon atoms
    [0, 2, 4, 2, 0, 6],   # Hydrogen atoms
    [2, 0, 1, 1, 1, 1]    # Oxygen atoms
], dtype=float)

atom_balance_check = atom_matrix @ N_stoich

print("\nAtom balance check:")
print(atom_balance_check)

if np.allclose(atom_balance_check, 0):
    print("Atom balance is satisfied for all reactions.")
else:
    print("Atom balance error detected.")
# #%%
# # Load thermodynamic data
# 
# # define function with arguments temperature and name of component
# def prop_thermo(T, comp):
#     results = np.empty(3) # generate empty vector of required size
#     t=T/1000 # define the dimensionless temperature required
#     # define the Shomate equations
#     fun_cp_NIST = np.array([t**0, t, t**2, t**3, t**-2, 0, 0, 0])
#     fun_H_NIST = np.array([t, t**2/2, t**3/3, t**4/4, -t**-1, 1, 0, -1])
#     fun_S_NIST = np.array([np.log(t), t, t**2/2, t**3/3, -t**-2/2, 0, 1, 0])
#     # assign parameters of the Shomate equation to components
#     if comp == 0 or comp == 'CO2': #
#         PhysProp_param_NIST = np.array([24.997, 55.187, -33.691, 7.948, -0.137, -403.608, 228.243, -393.522])
#     elif comp == 1 or comp == 'H2': #
#         PhysProp_param_NIST = np.array([33.07, -11.36, 11.43, -2.773, -0.1586, -9.981, 172.7, 0])
#     elif comp == 2 or comp == 'CH3OH': # No NIST data aviable, CoolProp data is used for comparison Max Temperature is 620K
#         PhysProp_param_NIST = np.array([])
#     elif comp == 3 or comp == 'H2O': #
#         PhysProp_param_NIST = np.array([30.092, 6.832, 6.793, -2.534, 0.082, -250.881, 223.397, -241.826])
#     elif comp == 4 or comp == 'CO': #
#         PhysProp_param_NIST = np.array([25.567, 6.096, 4.054, -2.671, 0.131, -483.607, 263.612, -110.527])
#     elif comp == 5 or comp == 'CH3OCH3': # No NIST data aviable, CoolProp data is used for comparison Max Temperature is 585K
#         PhysProp_param_NIST = np.array([])
#     else:
#         PhysProp_param_NIST = np.array([0, 0, 0, 0, 0, 0, 0, 0])
#     # multiply set of parameters to Shomate equation
#     results[0] = fun_cp_NIST.dot(PhysProp_param_NIST) # heat capacity
#     results[1] = (PhysProp_param_NIST[-1] + fun_H_NIST.dot(PhysProp_param_NIST))*1000 # enthalpy of formation
#     results[2] = fun_S_NIST.dot(PhysProp_param_NIST) # entropy
#     return results
# # call of the function for T = 25°C and CO2
# print('vector of heat capacity, enthalpy and entropy for carbon dioxide:',prop_thermo(25+273, 'CO2')) # display of key result for check

#%%

# Reference values at 298.15 K from NIST Webbook
# Methanol (CH3OH)
Hf_ref_MeOH  = -200_700   # J/mol   standard enthalpy of formation
S_ref_MeOH   =  239.9     # J/(mol·K) standard molar entropy

# Dimethyl ether (CH3OCH3 / DME)
Hf_ref_DME   = -184_100   # J/mol
S_ref_DME    =  267.8     # J/(mol·K)


T_REF = 298.15     # K,  NIST standard reference temperature

def _coolprop_shifted(T, fluid, Hf_ref, S_ref):
    """
    Return (Cp [J/mol/K], H [J/mol], S [J/mol/K]) for a CoolProp fluid,
    shifted so that the reference state matches NIST conventions:
      H(T_REF) = Hf_ref   [J/mol]
      S(T_REF) = S_ref     [J/mol/K]
    """
    Cp_T    = PropsSI('Cpmolar', 'T', T,     'P', 101325, fluid)   # J/mol/K
    H_T     = PropsSI('Hmolar',  'T', T,     'P', 101325, fluid)   # J/mol
    H_ref   = PropsSI('Hmolar',  'T', T_REF, 'P', 101325, fluid)   # J/mol  (CP baseline)
    S_T     = PropsSI('Smolar',  'T', T,     'P', 101325, fluid)   # J/mol/K
    S_ref_CP= PropsSI('Smolar',  'T', T_REF, 'P', 101325, fluid)   # J/mol/K (CP baseline)

    # --- shift to NIST reference ---
    H_shifted = Hf_ref + (H_T - H_ref)       # J/mol
    S_shifted = S_ref  + (S_T - S_ref_CP)    # J/mol/K

    return np.array([Cp_T, H_shifted, S_shifted])


def prop_thermo(T, comp):
    results = np.empty(3)
    t = T / 1000

    fun_cp_NIST = np.array([t**0, t, t**2, t**3, t**-2, 0, 0, 0])
    fun_H_NIST  = np.array([t, t**2/2, t**3/3, t**4/4, -t**-1, 1, 0, -1])
    fun_S_NIST  = np.array([np.log(t), t, t**2/2, t**3/3, -t**-2/2, 0, 1, 0])

    if comp == 0 or comp == 'CO2':
        p = np.array([24.997, 55.187, -33.691, 7.948, -0.137, -403.608, 228.243, -393.522])
    elif comp == 1 or comp == 'H2':
        p = np.array([33.07, -11.36, 11.43, -2.773, -0.1586, -9.981, 172.7, 0])
    elif comp == 3 or comp == 'H2O':
        p = np.array([30.092, 6.832, 6.793, -2.534, 0.082, -250.881, 223.397, -241.826])
    elif comp == 4 or comp == 'CO':
        p = np.array([25.567, 6.096, 4.054, -2.671, 0.131, -483.607, 263.612, -110.527])
    else:
        p = None

    if p is not None:
        results[0] = fun_cp_NIST.dot(p)
        results[1] = (p[-1] + fun_H_NIST.dot(p)) * 1000
        results[2] = fun_S_NIST.dot(p)
        return results

    # --- CoolProp path (with T limits) ---
    if comp == 2 or comp == 'CH3OH':
        if T > 620:
            raise ValueError(f"T={T} K exceeds CoolProp limit (620 K) for Methanol")
        return _coolprop_shifted(T, 'Methanol', Hf_ref_MeOH, S_ref_MeOH)

    elif comp == 5 or comp == 'CH3OCH3':
        if T > 585:
            raise ValueError(f"T={T} K exceeds CoolProp limit (585 K) for DME")
        return _coolprop_shifted(T, 'DimethylEther', Hf_ref_DME, S_ref_DME)

    else:
        return np.zeros(3)


# --- Sanity checks ---
print('CO2  @ 298 K :', prop_thermo(298.15, 'CO2'))
print('MeOH @ 298 K :', prop_thermo(298.15, 'CH3OH'))   # H ≈ -200700, S ≈ 239.9
print('DME  @ 298 K :', prop_thermo(298.15, 'CH3OCH3'))  # H ≈ -184100, S ≈ 267.8
print('MeOH @ 500 K :', prop_thermo(500,    'CH3OH'))
print('DME  @ 500 K :', prop_thermo(500,    'CH3OCH3'))

#%%

# define function with argument temperature
def rxn_data_1(T):
    res = np.empty(5) # generate empty vector of required size
    # call the function on thermodynamic properties for all components
    data_CO2 = prop_thermo(T, 'CO2')
    data_H2 = prop_thermo(T, 'H2')
    data_CH3OH = prop_thermo(T, 'CH3OH')
    data_H2O = prop_thermo(T, 'H2O')
    # use stoichiometry to calculate the thermodynamic properties of the reaction
    res[0] = data_CH3OH[0] + data_H2O[0] - 3*data_H2[0] - data_CO2[0] # reaction heat capacity
    res[1] = data_CH3OH[1] + data_H2O[1] - 3*data_H2[1] - data_CO2[1] # reaction enthalpy
    res[2] = data_CH3OH[2] + data_H2O[2] - 3*data_H2[2] - data_CO2[2] # reaction entropy
    res[3] = res[1] - T*res[2] # reaction Gibbs enthalpy
    res[4] = np.exp(-res[3]/(8.3145*T)) # reaction equilibrium constant
    return res
# call the function for T = 25°C to compare with tabulated data
print('vector with heat capacity, enthalpy, entropie and Gibbs enthalpy of the reaction, and thermodynamic equilibrium constant:',rxn_data_1(25+273)) # display of key result for check

# define function with argument temperature
def rxn_data_2(T):
    res = np.empty(5) # generate empty vector of required size
    # call the function on thermodynamic properties for all components
    data_CO = prop_thermo(T, 'CO')
    data_H2 = prop_thermo(T, 'H2')
    data_CH3OH = prop_thermo(T, 'CH3OH')
    # use stoichiometry to calculate the thermodynamic properties of the reaction
    res[0] = data_CH3OH[0] - 2*data_H2[0] - data_CO[0] # reaction heat capacity
    res[1] = data_CH3OH[1] - 2*data_H2[1] - data_CO[1] # reaction enthalpy
    res[2] = data_CH3OH[2] - 2*data_H2[2] - data_CO[2] # reaction entropy
    res[3] = res[1] - T*res[2] # reaction Gibbs enthalpy
    res[4] = np.exp(-res[3]/(8.3145*T)) # reaction equilibrium constant
    return res
# call the function for T = 25°C to compare with tabulated data
print('vector with heat capacity, enthalpy, entropie and Gibbs enthalpy of the reaction, and thermodynamic equilibrium constant:',rxn_data_2(25+273)) # display of key result for check

# define function with argument temperature
def rxn_data_3(T):
    res = np.empty(5) # generate empty vector of required size
    # call the function on thermodynamic properties for all components
    data_CH3OH = prop_thermo(T, 'CH3OH')
    data_H2O = prop_thermo(T, 'H2O')
    data_CH3OCH3 = prop_thermo(T, 'CH3OCH3')
    # use stoichiometry to calculate the thermodynamic properties of the reaction
    res[0] = data_CH3OCH3[0] + data_H2O[0] - 2* data_CH3OH[0]  # reaction heat capacity
    res[1] = data_CH3OCH3[1] + data_H2O[1] - 2* data_CH3OH[1]  # reaction enthalpy
    res[2] = data_CH3OCH3[2] + data_H2O[2] - 2* data_CH3OH[2]  # reaction entropy
    res[3] = res[1] - T*res[2] # reaction Gibbs enthalpy
    res[4] = np.exp(-res[3]/(8.3145*T)) # reaction equilibrium constant
    return res
# call the function for T = 25°C to compare with tabulated data
print('vector with heat capacity, enthalpy, entropie and Gibbs enthalpy of the reaction, and thermodynamic equilibrium constant:',rxn_data_3(25+273)) # display of key result for check


#%%
# define vectors for typical ranges of T and p
T = np.linspace(100+273, 200+273, 100) # temperature in K
p = np.array([1, 2, 3]) # total pressure in bar

K_x_1 = np.empty([p.shape[0],T.shape[0]]) # generate empty vector of required size
for TT in range(T.shape[0]): # vary the temperature within the given range
    # at each single temperature
    rxn_data_get = rxn_data_1(T[TT]) # call the function for thermodynamic data of the reaction 
    K_eq = rxn_data_get[-1] # the last element in the results vector provides the thermodynamic equilibrium constant
    K_x_1[:,TT] = K_eq*np.power(p,2) # calculate K_x at each temperature for all pressures

# generate the plot
plt.figure(figsize=(5, 5))
plt.grid()
plt.xlabel(r"$T\,/\,°C$")
plt.ylabel("$K_x\,/\,1$")
plt.plot(T[:]-273, K_x_1[0,:], 'r-', label="$10$ bar")
plt.plot(T[:]-273, K_x_1[1,:], 'g-', label="$20$ bar")
plt.plot(T[:]-273, K_x_1[2,:], 'b-', label="$30$ bar")
plt.legend(loc='best')
plt.show()

K_x_2 = np.empty([p.shape[0],T.shape[0]]) # generate empty vector of required size
for TT in range(T.shape[0]): # vary the temperature within the given range
    # at each single temperature
    rxn_data_get = rxn_data_2(T[TT]) # call the function for thermodynamic data of the reaction 
    K_eq = rxn_data_get[-1] # the last element in the results vector provides the thermodynamic equilibrium constant
    K_x_2[:,TT] = K_eq*np.power(p,2) # calculate K_x at each temperature for all pressures

# generate the plot
plt.figure(figsize=(5, 5))
plt.grid()
plt.xlabel(r"$T\,/\,°C$")
plt.ylabel("$K_x\,/\,1$")
plt.plot(T[:]-273, K_x_2[0,:], 'r-', label="$10$ bar")
plt.plot(T[:]-273, K_x_2[1,:], 'g-', label="$20$ bar")
plt.plot(T[:]-273, K_x_2[2,:], 'b-', label="$30$ bar")
plt.legend(loc='best')
plt.show()

K_x_3 = np.empty([p.shape[0],T.shape[0]]) # generate empty vector of required size
for TT in range(T.shape[0]): # vary the temperature within the given range
    # at each single temperature
    rxn_data_get = rxn_data_3(T[TT]) # call the function for thermodynamic data of the reaction 
    K_eq = rxn_data_get[-1] # the last element in the results vector provides the thermodynamic equilibrium constant
    K_x_3[:,TT] = K_eq*np.power(p,2) # calculate K_x at each temperature for all pressures

# generate the plot
plt.figure(figsize=(5, 5))
plt.grid()
plt.xlabel(r"$T\,/\,°C$")
plt.ylabel("$K_x\,/\,1$")
plt.plot(T[:]-273, K_x_3[0,:], 'r-', label="$10$ bar")
plt.plot(T[:]-273, K_x_3[1,:], 'g-', label="$20$ bar")
plt.plot(T[:]-273, K_x_3[2,:], 'b-', label="$30$ bar")
plt.legend(loc='best')
plt.show()
