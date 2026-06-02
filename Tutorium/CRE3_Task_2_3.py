#%%
# Import
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from numpy.linalg import matrix_rank


#%%
# Stoichiometric matrix

# Reactions:
# 1: CO  + 3H2 <=> CH4 + H2O
# 2: CO2 + 4H2 <=> CH4 + 2H2O
# 3: CO2 + H2  <=> CO  + H2O

N_stoich = np.array([[-1,  0,  -1],  # CO
                     [ 0, -1,   1],  # CO2
                     [ 1,  1,   0],  # CH4
                     [-3, -4,   1],  # H2
                     [ 1,  2,  -1]]) # H2O

# rank
N_stoich_rank = matrix_rank(N_stoich)

print('Rank:', N_stoich_rank)

# submatrices
N_stoich_11 = N_stoich[0:N_stoich_rank, 0:N_stoich_rank]
N_stoich_21 = N_stoich[N_stoich_rank:N_stoich.shape[0], 0:N_stoich_rank]
N_stoich_12 = N_stoich[0:N_stoich_rank, N_stoich_rank:N_stoich.shape[1]]
N_stoich_22 = N_stoich[N_stoich_rank:N_stoich.shape[0], N_stoich_rank:N_stoich.shape[1]]


#%%
# CoolProp thermodynamic properties

def prop_thermo(T, comp, P=101325):

    results = np.empty(3)

    fluid_map = {
        'H2':  'Hydrogen',
        'CO':  'CarbonMonoxide',
        'CO2': 'CarbonDioxide',
        'CH4': 'Methane',
        'H2O': 'Water'
    }

    # integer indexing
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

    fluid = fluid_map[comp]

    # ideal-gas heat capacity
    results[0] = PropsSI(
        'CP0MOLAR',
        'T', T,
        'P', P,
        fluid
    )

    # molar enthalpy
    results[1] = PropsSI(
        'HMOLAR',
        'T', T,
        'P', P,
        fluid
    )

    # molar entropy
    results[2] = PropsSI(
        'SMOLAR',
        'T', T,
        'P', P,
        fluid
    )

    return results


#%%
# Example property call

T = 25 + 273.15

print(
    'CO thermodynamic properties:',
    prop_thermo(T, 'CO')
)


#%%
# Methanation reaction thermodynamics
#
# CO2 + 4 H2 <=> CH4 + 2 H2O
#

def rxn_data_reaction1(T):

    res = np.empty(5)

    data_CO  = prop_thermo(T, 'CO')
    data_H2  = prop_thermo(T, 'H2')
    data_CH4 = prop_thermo(T, 'CH4')
    data_H2O = prop_thermo(T, 'H2O')

    # ΔCp
    res[0] = (
        data_CH4[0]
        + data_H2O[0]
        - data_CO[0]
        - 3*data_H2[0]
    )

    # ΔH
    res[1] = (
        data_CH4[1]
        + data_H2O[1]
        - data_CO[1]
        - 3*data_H2[1]
    )

    # ΔS
    res[2] = (
        data_CH4[2]
        + data_H2O[2]
        - data_CO[2]
        - 3*data_H2[2]
    )

    # ΔG
    res[3] = res[1] - T * res[2]

    # equilibrium constant
    res[4] = np.exp(
        -res[3] / (8.314462618 * T)
    )

    return res


#%%
# -------------------------------------------------
# Reaction 2
# CO2 + 4H2 <=> CH4 + 2H2O
# -------------------------------------------------

def rxn_data_reaction2(T):

    res = np.empty(5)

    data_CO2 = prop_thermo(T, 'CO2')
    data_H2  = prop_thermo(T, 'H2')
    data_CH4 = prop_thermo(T, 'CH4')
    data_H2O = prop_thermo(T, 'H2O')

    # ΔCp
    res[0] = (
        data_CH4[0]
        + 2*data_H2O[0]
        - data_CO2[0]
        - 4*data_H2[0]
    )

    # ΔH
    res[1] = (
        data_CH4[1]
        + 2*data_H2O[1]
        - data_CO2[1]
        - 4*data_H2[1]
    )

    # ΔS
    res[2] = (
        data_CH4[2]
        + 2*data_H2O[2]
        - data_CO2[2]
        - 4*data_H2[2]
    )

    # ΔG
    res[3] = res[1] - T * res[2]

    # equilibrium constant
    res[4] = np.exp(
        -res[3] / (8.314462618 * T)
    )

    return res


#%%
# -------------------------------------------------
# Reaction 3
# CO2 + H2 <=> CO + H2O
# -------------------------------------------------

def rxn_data_reaction3(T):

    res = np.empty(5)

    data_CO2 = prop_thermo(T, 'CO2')
    data_H2  = prop_thermo(T, 'H2')
    data_CO  = prop_thermo(T, 'CO')
    data_H2O = prop_thermo(T, 'H2O')

    # ΔCp
    res[0] = (
        data_CO[0]
        + data_H2O[0]
        - data_CO2[0]
        - data_H2[0]
    )

    # ΔH
    res[1] = (
        data_CO[1]
        + data_H2O[1]
        - data_CO2[1]
        - data_H2[1]
    )

    # ΔS
    res[2] = (
        data_CO[2]
        + data_H2O[2]
        - data_CO2[2]
        - data_H2[2]
    )

    # ΔG
    res[3] = res[1] - T * res[2]

    # equilibrium constant
    res[4] = np.exp(
        -res[3] / (8.314462618 * T)
    )

    return res


#%%
# Example calculations

T = 25 + 273.15

print('Reaction 1:')
print(rxn_data_reaction1(T))

print('\nReaction 2:')
print(rxn_data_reaction2(T))

print('\nReaction 3:')
print(rxn_data_reaction3(T))

#%%
# Temperature and pressure ranges

T = np.linspace(100 + 273.15, 200 + 273.15, 100)
p = np.array([100, 200, 300])   # bar

K_x_1 = np.empty((p.shape[0], T.shape[0]))
K_x_2 = np.empty((p.shape[0], T.shape[0]))
K_x_3 = np.empty((p.shape[0], T.shape[0]))


#%%
# Calculate equilibrium constants

for TT in range(T.shape[0]):

    rxn_1 = rxn_data_reaction1(T[TT])

    K_eq_1 = rxn_1[-1]

    # Δν = -2
    K_x_1[:, TT] = K_eq_1 * np.power(p, 2)

    rxn_2 = rxn_data_reaction2(T[TT])
    K_eq_2 = rxn_2[-1]
    K_x_2[:, TT] = K_eq_2 * np.power(p, 2)

    rxn_3 = rxn_data_reaction3(T[TT])
    K_eq_3 = rxn_3[-1] 
    K_x_3[:, TT] = K_eq_3 * np.power(p, 0)


#%%
# Plot

plt.figure(figsize=(5, 5))

plt.grid()

plt.xlabel(r"$T \, / \, ^\circ C$")
plt.ylabel(r"$K_x$")

plt.plot(T - 273.15, K_x_1[0, :], 'r-', label='100 bar')
plt.plot(T - 273.15, K_x_1[1, :], 'g-', label='200 bar')
plt.plot(T - 273.15, K_x_1[2, :], 'b-', label='300 bar')

plt.legend(loc='best')

plt.show()


plt.figure(figsize=(5, 5))

plt.grid()

plt.xlabel(r"$T \, / \, ^\circ C$")
plt.ylabel(r"$K_x$")

plt.plot(T - 273.15, K_x_2[0, :], 'r-', label='100 bar')
plt.plot(T - 273.15, K_x_2[1, :], 'g-', label='200 bar')
plt.plot(T - 273.15, K_x_2[2, :], 'b-', label='300 bar')

plt.legend(loc='best')

plt.show()

plt.figure(figsize=(5, 5))

plt.grid()

plt.xlabel(r"$T \, / \, ^\circ C$")
plt.ylabel(r"$K_x$")

plt.plot(T - 273.15, K_x_3[0, :], 'r-', label='100 bar')
plt.plot(T - 273.15, K_x_3[1, :], 'g-', label='200 bar')
plt.plot(T - 273.15, K_x_3[2, :], 'b-', label='300 bar')

plt.legend(loc='best')

plt.show()