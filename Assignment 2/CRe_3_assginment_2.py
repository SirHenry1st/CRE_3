#%%
# Imports of packages
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import ICIW_Plots.colors as iciw_colors
from CoolProp.CoolProp import PropsSI


# Load thermodynamic data

# define function with arguments temperature and name of component
def prop_thermo(T, comp):
    results = np.empty(3) # generate empty vector of required size
    t=T/1000 # define the dimensionless temperature required
    # define the Shomate equations
    fun_cp_NIST = np.array([t**0, t, t**2, t**3, t**-2, 0, 0, 0])
    fun_H_NIST = np.array([t, t**2/2, t**3/3, t**4/4, -t**-1, 1, 0, -1])
    fun_S_NIST = np.array([np.log(t), t, t**2/2, t**3/3, -t**-2/2, 0, 1, 0])
    # assign parameters of the Shomate equation to components
    if comp == 0 or comp == 'CO2': #
        PhysProp_param_NIST = np.array([24.997, 55.187, -33.691, 7.948, -0.137, -403.608, 228.243, -393.522])
    elif comp == 1 or comp == 'H2': #
        PhysProp_param_NIST = np.array([33.07, -11.36, 11.43, -2.773, -0.1586, -9.981, 172.7, 0])
    elif comp == 2 or comp == 'CH3OH': # No NIST data aviable, CoolProp data is used for comparison Max Temperature is 620K
        PhysProp_param_NIST = np.array([])
    elif comp == 3 or comp == 'H2O': #
        PhysProp_param_NIST = np.array([30.092, 6.832, 6.793, -2.534, 0.082, -250.881, 223.397, -241.826])
    elif comp == 4 or comp == 'CO': #
        PhysProp_param_NIST = np.array([25.567, 6.096, 4.054, -2.671, 0.131, -483.607, 263.612, -110.527])
    elif comp == 5 or comp == 'CH3OCH3': # No NIST data aviable, CoolProp data is used for comparison Max Temperature is 585K
        PhysProp_param_NIST = np.array([])
    else:
        PhysProp_param_NIST = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    # multiply set of parameters to Shomate equation
    results[0] = fun_cp_NIST.dot(PhysProp_param_NIST) # heat capacity
    results[1] = (PhysProp_param_NIST[-1] + fun_H_NIST.dot(PhysProp_param_NIST))*1000 # enthalpy of formation
    results[2] = fun_S_NIST.dot(PhysProp_param_NIST) # entropy
    return results
# call of the function for T = 25°C and CO2
print('vector of heat capacity, enthalpy and entropy for carbon dioxide:',prop_thermo(25+273, 'CO2')) # display of key result for check

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
