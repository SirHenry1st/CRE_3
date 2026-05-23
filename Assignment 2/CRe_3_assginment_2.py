#%%
# Imports of packages
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import ICIW_Plots.colors as iciw_colors
from CoolProp.CoolProp import PropsSI
from numpy.linalg import matrix_rank
from scipy.optimize import fsolve # import function for root finding
from scipy.optimize import root ## import of the numerical solver function


#%%
# Stochimetric analysis

# Define components and reactions
components = np.array(["CO2", "H2", "CH3OH", "H2O", "CO", "DME"])

reactions = np.array([
    "R1: CO2 + 3H2 -> CH3OH + H2O",
    "R2: CO + 2H2 -> CH3OH",
    "R3: CO2 + H2 -> CO + H2O",
    "R4: 2CH3OH -> DME + H2O"
])
# Stoichiometric matrix
# Rows = components
# Columns = reactions

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

# Rank of the stoichiometric matrix
rank_N = matrix_rank(N_stoich)
print("\nRank of full stoichiometric matrix:", rank_N)

# Check dependency of R3
print("\nCheck stoichiometric dependency:")
print("R3 column:")
print(N_stoich[:, 2])

print("\nR1 - R2 column:")
print(N_stoich[:, 0] - N_stoich[:, 1])

print("\nIs R3 = R1 - R2?")
print(np.allclose(N_stoich[:, 2], N_stoich[:, 0] - N_stoich[:, 1]))

# Choose key reactions and key components
# Key reactions: R1, R2, R4
# Key components: CO2, CO, DME

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

# Reaction extent calculation example

# Use only independent reactions R1, R2 and R4
N_independent = N_stoich[:, key_reaction_idx]

# Example inlet molar flow rates in mol/s
# Component order: CO2, H2, CH3OH, H2O, CO, DME
n_in = np.array([2.0, 6.0, 0.0, 0.0, 1.0, 0.0])

# Example measured changes of key components
# CO2 consumed by 0.8 mol/s
# CO consumed by 0.4 mol/s
# DME formed by 0.3 mol/s
Delta_n_key = np.array([-0.8, -0.4, 0.3])

# Solve reaction extent
xi_independent = np.linalg.solve(N11, Delta_n_key)

print("\nIndependent reaction extents in mol/s:")

for reaction, extent in zip(reactions[key_reaction_idx], xi_independent):
    print(f"{reaction}: {extent:.4f} mol/s")

# Calculate change in all components
Delta_n_all = N_independent @ xi_independent

# Calculate outlet molar flow rates
n_out = n_in + Delta_n_all

print("\nChanges in molar flow rates in mol/s:")

for component, change in zip(components, Delta_n_all):
    print(f"{component}: {change:.4f}")

print("\nOutlet molar flow rates in mol/s:")

for component, flow in zip(components, n_out):
    print(f"{component}: {flow:.4f}")

# DME carbon yield based on total carbon feed from CO2 and CO
carbon_feed = n_in[0] + n_in[4]

Y_DME_carbon = (2 * n_out[5]) / carbon_feed

print("\nDME carbon yield:")
print(f"{Y_DME_carbon:.4f}")

# Verification using atom balance
# Atom matrix order: C, H, O
# Component order: CO2, H2, CH3OH, H2O, CO, DME

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
        p = np.array([25.568, 6.096, 4.054, -2.671, 0.131, -118.009, 227.367, -110.527])
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
print('CO  @ 298 K :', prop_thermo(298.15, 'CO'))
print('MeOH @ 298 K :', prop_thermo(298.15, 'CH3OH'))   # H ≈ -200700, S ≈ 239.9
print('DME  @ 298 K :', prop_thermo(298.15, 'CH3OCH3'))  # H ≈ -184100, S ≈ 267.8
print('MeOH @ 500 K :', prop_thermo(500,    'CH3OH'))
print('DME  @ 500 K :', prop_thermo(500,    'CH3OCH3'))
print('CO  @ 500 K :', prop_thermo(500,    'CO'))

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

def rxn_data_3(T):
    """R3: CO2 + H2 <-> CO + H2O  (reverse water-gas shift)"""
    res = np.empty(5)
    data_CO2   = prop_thermo(T, 'CO2')
    data_H2    = prop_thermo(T, 'H2')
    data_CO    = prop_thermo(T, 'CO')
    data_H2O   = prop_thermo(T, 'H2O')
    res[0] = data_CO[0]  + data_H2O[0] - data_H2[0]  - data_CO2[0]
    res[1] = data_CO[1]  + data_H2O[1] - data_H2[1]  - data_CO2[1]
    res[2] = data_CO[2]  + data_H2O[2] - data_H2[2]  - data_CO2[2]
    res[3] = res[1] - T * res[2]
    res[4] = np.exp(-res[3] / (8.3145 * T))
    return res
print('vector with heat capacity, enthalpy, entropie and Gibbs enthalpy of the reaction, and thermodynamic equilibrium constant:',rxn_data_3(25+273)) # display of key result for check

def rxn_data_4(T):
    """R4: 2CH3OH <-> DME + H2O"""
    res = np.empty(5)
    data_CH3OH   = prop_thermo(T, 'CH3OH')
    data_H2O     = prop_thermo(T, 'H2O')
    data_DME     = prop_thermo(T, 'CH3OCH3')
    res[0] = data_DME[0]  + data_H2O[0] - 2*data_CH3OH[0]
    res[1] = data_DME[1]  + data_H2O[1] - 2*data_CH3OH[1]
    res[2] = data_DME[2]  + data_H2O[2] - 2*data_CH3OH[2]
    res[3] = res[1] - T * res[2]
    res[4] = np.exp(-res[3] / (8.3145 * T))
    return res
print('vector with heat capacity, enthalpy, entropie and Gibbs enthalpy of the reaction, and thermodynamic equilibrium constant:',rxn_data_4(25+273)) # display of key result for check


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
    K_x_2[:,TT] = K_eq*np.power(p,1.5) # calculate K_x at each temperature for all pressures

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
    K_x_3[:,TT] = K_eq*np.power(p,1.5) # calculate K_x at each temperature for all pressures

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

#%%

# Exponents for K_x = K° * (p/p°)^(-sum_nu)
# sum_nu per reaction: R1=-2, R2=-2, R3=0, R4=0
# => K_x exponent:     R1=+2, R2=+2, R3=0, R4=0

T = np.linspace(100+273, 200+273, 100)  # K
p = np.array([10, 20, 30])              # bar  ← fix: your labels said 10/20/30 bar
p_std = 1                               # bar  (standard pressure p°)

# --- R1: CO2 + 3H2 <-> CH3OH + H2O,  sum_nu = -2 ---
K_x_1 = np.empty([p.shape[0], T.shape[0]])
for TT in range(T.shape[0]):
    K_eq = rxn_data_1(T[TT])[-1]
    K_x_1[:, TT] = K_eq * (p / p_std)**2   # exponent = -sum_nu = +2

plt.figure(figsize=(5, 5))
plt.grid()
plt.title("R1: CO₂ + 3H₂ ⇌ CH₃OH + H₂O")
plt.xlabel(r"$T\,/\,°C$")
plt.ylabel(r"$K_x$")
plt.plot(T - 273, K_x_1[0, :], 'r-', label="10 bar")
plt.plot(T - 273, K_x_1[1, :], 'g-', label="20 bar")
plt.plot(T - 273, K_x_1[2, :], 'b-', label="30 bar")
plt.legend()
plt.tight_layout()
plt.show()

# --- R2: CO + 2H2 <-> CH3OH,  sum_nu = -2 ---
K_x_2 = np.empty([p.shape[0], T.shape[0]])
for TT in range(T.shape[0]):
    K_eq = rxn_data_2(T[TT])[-1]
    K_x_2[:, TT] = K_eq * (p / p_std)**2   # exponent = -sum_nu = +2

plt.figure(figsize=(5, 5))
plt.grid()
plt.title("R2: CO + 2H₂ ⇌ CH₃OH")
plt.xlabel(r"$T\,/\,°C$")
plt.ylabel(r"$K_x$")
plt.plot(T - 273, K_x_2[0, :], 'r-', label="10 bar")
plt.plot(T - 273, K_x_2[1, :], 'g-', label="20 bar")
plt.plot(T - 273, K_x_2[2, :], 'b-', label="30 bar")
plt.legend()
plt.tight_layout()
plt.show()

# --- R3: CO2 + H2 <-> CO + H2O,  sum_nu = 0 → pressure independent ---
K_x_3 = np.empty([p.shape[0], T.shape[0]])
for TT in range(T.shape[0]):
    K_eq = rxn_data_3(T[TT])[-1]           # no rxn_data_3 yet — see note below
    K_x_3[:, TT] = K_eq * np.ones(p.shape) # (p/p°)^0 = 1

plt.figure(figsize=(5, 5))
plt.grid()
plt.title("R3: CO₂ + H₂ ⇌ CO + H₂O  (pressure-independent)")
plt.xlabel(r"$T\,/\,°C$")
plt.ylabel(r"$K_x$")
plt.plot(T - 273, K_x_3[0, :], 'r-', label="10 bar")
plt.plot(T - 273, K_x_3[1, :], 'g-', label="20 bar")
plt.plot(T - 273, K_x_3[2, :], 'b-', label="30 bar")
plt.legend()
plt.tight_layout()
plt.show()

# --- R4: 2CH3OH <-> DME + H2O,  sum_nu = 0 → pressure independent ---
K_x_4 = np.empty([p.shape[0], T.shape[0]])
for TT in range(T.shape[0]):
    K_eq = rxn_data_4(T[TT])[-1]
    K_x_4[:, TT] = K_eq * np.ones(p.shape) # (p/p°)^0 = 1

plt.figure(figsize=(5, 5))
plt.grid()
plt.title("R4: 2CH₃OH ⇌ DME + H₂O  (pressure-independent)")
plt.xlabel(r"$T\,/\,°C$")
plt.ylabel(r"$K_x$")
plt.plot(T - 273, K_x_4[0, :], 'r-', label="10 bar")
plt.plot(T - 273, K_x_4[1, :], 'g-', label="20 bar")
plt.plot(T - 273, K_x_4[2, :], 'b-', label="30 bar")
plt.legend()
plt.tight_layout()
plt.show()

#%%
# Define function for calculation of outlet molar fractions with reaction extents
# and inlet molar flow rates as arguments
def composition(xi, n_in):
    """
    Calculate outlet molar fractions from reaction extents
    xi = [xi1, xi2, xi4]  - extents of reactions R1, R2, R4
    n_in = [n_CO2, n_H2, n_CH3OH, n_H2O, n_CO, n_DME]
    """
    xi1, xi2, xi4 = xi
    
    # Outlet molar flows (material balances)
    n_CO2   = n_in[0] - xi1
    n_H2    = n_in[1] - 3*xi1 - 2*xi2
    n_CH3OH = n_in[2] + xi1 + xi2 - 2*xi4
    n_H2O   = n_in[3] + xi1 + xi4
    n_CO    = n_in[4] - xi2
    n_DME   = n_in[5] + xi4
    
    # Total outlet flow (sum_nu: R1=-2, R2=-2, R4=0)
    n_out = np.sum(n_in) - 2*xi1 - 2*xi2
    
    # Molar fractions
    x = np.array([n_CO2, n_H2, n_CH3OH, n_H2O, n_CO, n_DME]) / n_out
    
    return x


# Define function for calculation of reaction extents
# This function describes the non-linear system of equations to be solved
def rxn_ext(xi, n_in, T, p):
    """
    System of equilibrium equations
    Returns residuals [f1, f2, f4] that should equal zero at equilibrium
    """
    # Get outlet composition
    x = composition(xi, n_in)
    x_CO2, x_H2, x_CH3OH, x_H2O, x_CO, x_DME = x
    if np.any(x < 0):
       return np.array([1e10, 1e10, 1e10])  # Penalize negative compositions to avoid unphysical solutions
    
    # Get thermodynamic data for all reactions
    rxn_data_get_1 = rxn_data_1(T)
    rxn_data_get_2 = rxn_data_2(T)
    rxn_data_get_4 = rxn_data_4(T)
    
    # Get thermodynamic equilibrium constants K°
    K1 = rxn_data_get_1[-1]
    K2 = rxn_data_get_2[-1]
    K4 = rxn_data_get_4[-1]
    
    # Convert to K_x (accounting for pressure dependency)
    p_std = 1  # bar
    Kx1 = K1 * np.power(p/p_std, 2)  # delta_nu = -2
    Kx2 = K2 * np.power(p/p_std, 2)  # delta_nu = -2
    Kx4 = K4 * np.power(p/p_std, 0)  # delta_nu = 0
    
    # Definition of non-linear equations for reaction extents
    # R1: CO2 + 3H2 <-> CH3OH + H2O
    res1 = Kx1 * np.power(x_CO2, 1) * np.power(x_H2, 3) - np.power(x_CH3OH, 1) * np.power(x_H2O, 1)
    
    # R2: CO + 2H2 <-> CH3OH
    res2 = Kx2 * np.power(x_CO, 1) * np.power(x_H2, 2) - np.power(x_CH3OH, 1)
    
    # R4: 2CH3OH <-> DME + H2O
    res4 = Kx4 * np.power(x_CH3OH, 2) - np.power(x_DME, 1) * np.power(x_H2O, 1)
    
    return [res1, res2, res4]


# Set inlet molar flow rates
n_in = np.array([2.0, 6.0, 0.0, 0.0, 0.0, 0.0])  # CO2, H2, CH3OH, H2O, CO, DME

# Define temperature and pressure ranges
T = np.linspace(100+273, 300+273, 30)  # K (start at 150°C for better convergence)
p = np.array([20, 50, 100])  # bar

# Generate empty arrays for storing results
# Shape: [n_pressures, n_temperatures, n_reactions]
xi = np.empty([p.shape[0], T.shape[0], 3])

print("Solving equilibrium for reaction network...")

for pp in range(p.shape[0]):  # vary pressure
    print(f"\nPressure: {p[pp]} bar")
    
    for TT in range(T.shape[0]):  # vary temperature
        # Initial guess for [xi1, xi2, xi4]
        xi_guess = np.array([0.5, 0.01, 0.1])
        
        # Call root finding algorithm
        # Variant via root (more robust for systems of equations)
        result = root(rxn_ext, xi_guess, args=(n_in, T[TT], p[pp]), method='hybr')
        
        if result.success:
            xi[pp, TT, :] = result.x
            
            if TT % 5 == 0:  # Print progress every 5 steps
                print(f"  T={T[TT]-273:.0f}°C: ξ=[{result.x[0]:.4f}, {result.x[1]:.4f}, {result.x[2]:.4f}]")
        else:
            print(f"  WARNING: Failed at T={T[TT]-273:.0f}°C")
            xi[pp, TT, :] = np.nan

# Plot results for each reaction
reaction_labels = ["R1: CO₂ + 3H₂ → CH₃OH + H₂O", 
                   "R2: CO + 2H₂ → CH₃OH",
                   "R4: 2CH₃OH → DME + H₂O"]

colors = ['r-', 'g-', 'b-']

for rxn_idx in range(3):
    plt.figure(figsize=(6, 5))
    plt.grid()
    plt.title(reaction_labels[rxn_idx])
    plt.xlabel(r"$T\,/\,°C$")
    plt.ylabel(r"$\xi\,/\,\mathrm{mol\,s^{-1}}$")
    
    plt.plot(T[:]-273, xi[0, :, rxn_idx], colors[0], label=f"{p[0]:.0f} bar", linewidth=2)
    plt.plot(T[:]-273, xi[1, :, rxn_idx], colors[1], label=f"{p[1]:.0f} bar", linewidth=2)
    plt.plot(T[:]-273, xi[2, :, rxn_idx], colors[2], label=f"{p[2]:.0f} bar", linewidth=2)
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Verification at one operating point
print("\n=== Verification: T=250°C, p=100 bar ===")
T_test = 250 + 273
p_test = 100

result = root(rxn_ext, [0.5, 0.01, 0.1], args=(n_in, T_test, p_test), method='hybr')

if result.success:
    xi_test = result.x
    print(f"Converged: {result.success}")
    print(f"\nReaction extents [mol/s]:")
    for i, label in enumerate(reaction_labels):
        print(f"  {label}: ξ = {xi_test[i]:.6f}")
    
    # Calculate outlet composition
    x_test = composition(xi_test, n_in)
    n_test = x_test * (np.sum(n_in) - 2*xi_test[0] - 2*xi_test[1])
    
    print(f"\nOutlet composition:")
    comp_names = ["CO2", "H2", "CH3OH", "H2O", "CO", "DME"]
    for i, name in enumerate(comp_names):
        print(f"  {name}: {n_test[i]:.4f} mol/s ({x_test[i]*100:.2f} mol%)")
else:
    print(f"Failed: {result.message}")