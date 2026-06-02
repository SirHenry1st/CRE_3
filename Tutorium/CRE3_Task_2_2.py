#%%
# Import

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank


#%%
# Stoichiometric matrix

# Reactions:
# 1: CO  + 3H2 <=> CH4 + H2O
# 2: CO2 + 4H2 <=> CH4 + 2H2O
# 3: CO2 + H2  <=> CO  + H2O

N_stoich = np.array([
    [-1,  0, -1],   # CO
    [ 0, -1,  1],   # CO2
    [ 1,  1,  0],   # CH4
    [-3, -4, -1],   # H2
    [ 1,  2,  1]    # H2O
])

# rank of stoichiometric matrix
N_stoich_rank = matrix_rank(N_stoich)

print('Rank:', N_stoich_rank)

# split matrices
N_stoich_11 = N_stoich[0:N_stoich_rank, 0:N_stoich_rank]
N_stoich_21 = N_stoich[N_stoich_rank:N_stoich.shape[0], 0:N_stoich_rank]
N_stoich_12 = N_stoich[0:N_stoich_rank, N_stoich_rank:N_stoich.shape[1]]
N_stoich_22 = N_stoich[N_stoich_rank:N_stoich.shape[0], N_stoich_rank:N_stoich.shape[1]]


#%%
# NIST Shomate thermodynamic properties

def prop_thermo(T, comp):

    results = np.empty(3)

    # dimensionless temperature
    t = T / 1000

    # Shomate basis functions
    fun_cp_NIST = np.array([
        t**0,
        t,
        t**2,
        t**3,
        t**-2,
        0,
        0,
        0
    ])

    fun_H_NIST = np.array([
        t,
        t**2 / 2,
        t**3 / 3,
        t**4 / 4,
        -t**-1,
        1,
        0,
        -1
    ])

    fun_S_NIST = np.array([
        np.log(t),
        t,
        t**2 / 2,
        t**3 / 3,
        -t**-2 / 2,
        0,
        1,
        0
    ])

    # species coefficients
    if comp == 0 or comp == 'H2':

        coeff = np.array([
            33.066178,
            -11.363417,
            11.432816,
            -2.772874,
            -0.158558,
            -9.980797,
            172.707974,
            0.0
        ])

    elif comp == 1 or comp == 'CO':

        coeff = np.array([
            25.56759,
            6.096130,
            4.054656,
            -2.671301,
            0.131021,
            -118.0089,
            227.3665,
            -110.5271
        ])

    elif comp == 2 or comp == 'CO2':

        coeff = np.array([
            24.99735,
            55.18696,
            -33.69137,
            7.948387,
            -0.136638,
            -403.6075,
            228.2431,
            -393.5224
        ])

    elif comp == 3 or comp == 'CH4':

        coeff = np.array([
            -0.703029,
            108.4773,
            -42.52157,
            5.862788,
            0.678565,
            -76.84376,
            158.7163,
            -74.87310
        ])

    elif comp == 4 or comp == 'H2O':

        coeff = np.array([
            30.09200,
            6.832514,
            6.793435,
            -2.534480,
            0.082139,
            -250.8810,
            223.3967,
            -241.8264
        ])

    else:

        coeff = np.zeros(8)

    # thermodynamic properties

    # Cp [J/mol/K]
    results[0] = fun_cp_NIST.dot(coeff)

    # H [J/mol]
    results[1] = (
        coeff[-1]
        + fun_H_NIST.dot(coeff)
    ) * 1000

    # S [J/mol/K]
    results[2] = fun_S_NIST.dot(coeff)

    return results


#%%
# Example component calculation

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

def rxn_data_methanation(T):

    res = np.empty(5)

    # component thermodynamic properties

    data_CO2 = prop_thermo(T, 'CO2')
    data_H2  = prop_thermo(T, 'H2')
    data_CH4 = prop_thermo(T, 'CH4')
    data_H2O = prop_thermo(T, 'H2O')

    # ΔCp [J/mol/K]
    res[0] = (
        data_CH4[0]
        + 2*data_H2O[0]
        - data_CO2[0]
        - 4*data_H2[0]
    )

    # ΔH [J/mol]
    res[1] = (
        data_CH4[1]
        + 2*data_H2O[1]
        - data_CO2[1]
        - 4*data_H2[1]
    )

    # ΔS [J/mol/K]
    res[2] = (
        data_CH4[2]
        + 2*data_H2O[2]
        - data_CO2[2]
        - 4*data_H2[2]
    )

    # ΔG [J/mol]
    res[3] = res[1] - T * res[2]

    # equilibrium constant
    res[4] = np.exp(
        -res[3] / (8.314462618 * T)
    )

    return res


#%%
# Example reaction calculation

print(
    'Reaction thermodynamics:',
    rxn_data_methanation(T)
)


#%%
# Temperature and pressure range

T = np.linspace(400 + 273.15, 500 + 273.15, 100)

p = np.array([
    100,
    200,
    300
])  # bar


#%%
# Equilibrium constants

K_x = np.empty((p.shape[0], T.shape[0]))

for TT in range(T.shape[0]):

    rxn = rxn_data_methanation(T[TT])

    K_eq = rxn[-1]

    # Δν = -2
    K_x[:, TT] = K_eq * np.power(p, 2)


#%%
# Plot

plt.figure(figsize=(5, 5))

plt.grid()

plt.xlabel(r"$T \, / \, ^\circ C$")
plt.ylabel(r"$K_x$")

plt.plot(
    T - 273.15,
    K_x[0, :],
    'r-',
    label='100 bar'
)

plt.plot(
    T - 273.15,
    K_x[1, :],
    'g-',
    label='200 bar'
)

plt.plot(
    T - 273.15,
    K_x[2, :],
    'b-',
    label='300 bar'
)

plt.legend(loc='best')

plt.show()