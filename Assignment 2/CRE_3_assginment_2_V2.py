#%%
# IMPORTS
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import ICIW_Plots.colors as iciw_colors
from CoolProp.CoolProp import PropsSI
from numpy.linalg import matrix_rank
from scipy.optimize import fsolve, root


#%%
# STOICHIOMETRIC ANALYSIS

# Define components and reactions
components = np.array(["CO2", "H2", "CH3OH", "H2O", "CO", "DME"])

reactions = np.array([
    "R1: CO2 + 3H2 -> CH3OH + H2O",
    "R2: CO + 2H2 -> CH3OH",
    "R3: CO2 + H2 -> CO + H2O",
    "R4: 2CH3OH -> DME + H2O"
])

# Stoichiometric matrix (rows = components, columns = reactions)
N_stoich = np.array([
    [-1,  0, -1,  0],   # CO2
    [-3, -2, -1,  0],   # H2
    [ 1,  1,  0, -2],   # CH3OH
    [ 1,  0,  1,  1],   # H2O
    [ 0, -1,  1,  0],   # CO
    [ 0,  0,  0,  1]    # DME
], dtype=float)

print("=" * 60)
print("STOICHIOMETRIC ANALYSIS")
print("=" * 60)
print("\nStoichiometric matrix N:")
print(N_stoich)

# Check rank and linear dependency
rank_N = matrix_rank(N_stoich)
print(f"\nRank of stoichiometric matrix: {rank_N}")
print(f"\nR3 = R1 - R2? {np.allclose(N_stoich[:, 2], N_stoich[:, 0] - N_stoich[:, 1])}")

# Define key reactions and key components
key_reaction_idx = [0, 1, 3]     # R1, R2, R4 (independent)
key_component_idx = [0, 4, 5]    # CO2, CO, DME

N11 = N_stoich[np.ix_(key_component_idx, key_reaction_idx)]
N_independent = N_stoich[:, key_reaction_idx]

print(f"\nKey reactions: {reactions[key_reaction_idx]}")
print(f"Key components: {components[key_component_idx]}")
print(f"\nReduced matrix N11:\n{N11}")
print(f"Rank of N11: {matrix_rank(N11)}")
print(f"\nKey reactions are valid: {matrix_rank(N11) == rank_N}")

# Verify atom balance
atom_matrix = np.array([
    [1, 0, 1, 0, 1, 2],   # Carbon
    [0, 2, 4, 2, 0, 6],   # Hydrogen
    [2, 0, 1, 1, 1, 1]    # Oxygen
], dtype=float)

atom_balance = atom_matrix @ N_stoich
print(f"\nAtom balance satisfied: {np.allclose(atom_balance, 0)}")


#%%
# THERMODYNAMIC PROPERTIES

# Reference values at 298.15 K from NIST
Hf_ref_MeOH  = -200_700   # J/mol
S_ref_MeOH   =  239.9     # J/(mol·K)
Hf_ref_DME   = -184_100   # J/mol
S_ref_DME    =  267.8     # J/(mol·K)
T_REF = 298.15            # K

def _coolprop_shifted(T, fluid, Hf_ref, S_ref):
    """Get thermodynamic properties from CoolProp, shifted to NIST reference."""
    Cp_T    = PropsSI('Cpmolar', 'T', T,     'P', 101325, fluid)
    H_T     = PropsSI('Hmolar',  'T', T,     'P', 101325, fluid)
    H_ref   = PropsSI('Hmolar',  'T', T_REF, 'P', 101325, fluid)
    S_T     = PropsSI('Smolar',  'T', T,     'P', 101325, fluid)
    S_ref_CP= PropsSI('Smolar',  'T', T_REF, 'P', 101325, fluid)
    
    H_shifted = Hf_ref + (H_T - H_ref)
    S_shifted = S_ref  + (S_T - S_ref_CP)
    
    return np.array([Cp_T, H_shifted, S_shifted])


def prop_thermo(T, comp):
    """
    Get thermodynamic properties: [Cp (J/mol/K), H (J/mol), S (J/mol/K)]
    Uses NIST Shomate equations for most species, CoolProp for CH3OH and DME.
    """
    results = np.empty(3)
    t = T / 1000  # Dimensionless temperature for Shomate

    # Shomate equation basis functions
    fun_cp_NIST = np.array([t**0, t, t**2, t**3, t**-2, 0, 0, 0])
    fun_H_NIST  = np.array([t, t**2/2, t**3/3, t**4/4, -t**-1, 1, 0, -1])
    fun_S_NIST  = np.array([np.log(t), t, t**2/2, t**3/3, -t**-2/2, 0, 1, 0])

    # NIST Shomate parameters
    params = {
        0: np.array([24.997, 55.187, -33.691, 7.948, -0.137, -403.608, 228.243, -393.522]),  # CO2
        1: np.array([33.07, -11.36, 11.43, -2.773, -0.1586, -9.981, 172.7, 0]),              # H2
        3: np.array([30.092, 6.832, 6.793, -2.534, 0.082, -250.881, 223.397, -241.826]),     # H2O
        4: np.array([25.568, 6.096, 4.054, -2.671, 0.131, -118.009, 227.367, -110.527])      # CO
    }
    
    # Map string names to indices
    if comp == 'CO2': comp = 0
    elif comp == 'H2': comp = 1
    elif comp == 'CH3OH': comp = 2
    elif comp == 'H2O': comp = 3
    elif comp == 'CO': comp = 4
    elif comp == 'CH3OCH3': comp = 5
    
    # Calculate properties
    if comp in params:
        p = params[comp]
        results[0] = fun_cp_NIST.dot(p)
        results[1] = (p[-1] + fun_H_NIST.dot(p)) * 1000
        results[2] = fun_S_NIST.dot(p)
    elif comp == 2:  # CH3OH
        if T > 620:
            raise ValueError(f"T={T} K exceeds CoolProp limit (620 K) for Methanol")
        results = _coolprop_shifted(T, 'Methanol', Hf_ref_MeOH, S_ref_MeOH)
    elif comp == 5:  # DME
        if T > 585:
            raise ValueError(f"T={T} K exceeds CoolProp limit (585 K) for DME")
        results = _coolprop_shifted(T, 'DimethylEther', Hf_ref_DME, S_ref_DME)
    else:
        results = np.zeros(3)
    
    return results


# Verification
print("\n" + "=" * 60)
print("THERMODYNAMIC PROPERTIES VERIFICATION (at 298.15 K)")
print("=" * 60)
for comp_name in ['CO2', 'H2', 'H2O', 'CO', 'CH3OH', 'CH3OCH3']:
    props = prop_thermo(298.15, comp_name)
    print(f"{comp_name:8s}: Cp={props[0]:7.2f} J/mol/K, H={props[1]:10.1f} J/mol, S={props[2]:6.2f} J/mol/K")


#%%
# REACTION THERMODYNAMICS

def rxn_data(T, reactants, products, stoich_r, stoich_p):
    """Generic function to calculate reaction properties."""
    res = np.empty(5)
    
    # Get properties for all species
    data_r = [prop_thermo(T, r) for r in reactants]
    data_p = [prop_thermo(T, p) for p in products]
    
    # Calculate reaction properties
    res[0] = sum(s*d[0] for s, d in zip(stoich_p, data_p)) - sum(s*d[0] for s, d in zip(stoich_r, data_r))  # Cp
    res[1] = sum(s*d[1] for s, d in zip(stoich_p, data_p)) - sum(s*d[1] for s, d in zip(stoich_r, data_r))  # H
    res[2] = sum(s*d[2] for s, d in zip(stoich_p, data_p)) - sum(s*d[2] for s, d in zip(stoich_r, data_r))  # S
    res[3] = res[1] - T * res[2]  # G
    res[4] = np.exp(-res[3] / (8.3145 * T))  # K
    
    return res


def rxn_data_1(T):
    """R1: CO2 + 3H2 <-> CH3OH + H2O"""
    return rxn_data(T, ['CO2', 'H2'], ['CH3OH', 'H2O'], [1, 3], [1, 1])


def rxn_data_2(T):
    """R2: CO + 2H2 <-> CH3OH"""
    return rxn_data(T, ['CO', 'H2'], ['CH3OH'], [1, 2], [1])


def rxn_data_3(T):
    """R3: CO2 + H2 <-> CO + H2O (reverse water-gas shift)"""
    return rxn_data(T, ['CO2', 'H2'], ['CO', 'H2O'], [1, 1], [1, 1])


def rxn_data_4(T):
    """R4: 2CH3OH <-> DME + H2O"""
    return rxn_data(T, ['CH3OH'], ['CH3OCH3', 'H2O'], [2], [1, 1])


# Verification at 298.15 K
print("\n" + "=" * 60)
print("REACTION THERMODYNAMICS (at 298.15 K)")
print("=" * 60)
for i, (rxn_name, rxn_func) in enumerate([
    ('R1', rxn_data_1), ('R2', rxn_data_2), ('R3', rxn_data_3), ('R4', rxn_data_4)
], 1):
    data = rxn_func(298.15)
    print(f"{rxn_name}: ΔH={data[1]/1000:7.1f} kJ/mol, ΔS={data[2]:7.2f} J/mol/K, ΔG={data[3]/1000:7.1f} kJ/mol, K={data[4]:.2e}")


#%%
# EQUILIBRIUM CONSTANTS vs T and p

def plot_Kx(T_range, p_range, rxn_func, rxn_name, delta_nu):
    """Plot equilibrium constant K_x vs temperature for different pressures."""
    K_x = np.empty([len(p_range), len(T_range)])
    p_std = 1  # bar
    
    for i, T in enumerate(T_range):
        K_eq = rxn_func(T)[-1]
        K_x[:, i] = K_eq * (p_range / p_std)**(-delta_nu)
    
    plt.figure(figsize=(6, 5))
    plt.grid()
    plt.title(f"{rxn_name}")
    plt.xlabel(r"$T\,/\,°C$")
    plt.ylabel(r"$K_x$")
    plt.yscale('log')
    
    colors = ['r-', 'g-', 'b-']
    for i, (p_val, color) in enumerate(zip(p_range, colors)):
        plt.plot(T_range - 273, K_x[i, :], color, label=f"{p_val:.0f} bar", linewidth=2)
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


# Calculate and plot K_x for all reactions
T_range = np.linspace(100+273, 200+273, 100)
p_range = np.array([10, 20, 30])

reactions_plot = [
    (rxn_data_1, "R1: CO₂ + 3H₂ ⇌ CH₃OH + H₂O", -2),
    (rxn_data_2, "R2: CO + 2H₂ ⇌ CH₃OH", -2),
    (rxn_data_3, "R3: CO₂ + H₂ ⇌ CO + H₂O", 0),
    (rxn_data_4, "R4: 2CH₃OH ⇌ DME + H₂O", 0)
]

for rxn_func, rxn_name, delta_nu in reactions_plot:
    plot_Kx(T_range, p_range, rxn_func, rxn_name, delta_nu)


#%%
# EQUILIBRIUM COMPOSITION SOLVER

def composition(xi, n_in):
    """Calculate outlet molar fractions from reaction extents."""
    xi1, xi2, xi4 = xi
    n_out = np.sum(n_in) - 2*xi1 - 2*xi2
    
    if n_out <= 1e-10:
        return np.full(6, np.nan)
    
    x = np.array([
        (n_in[0] - xi1) / n_out,              # CO2
        (n_in[1] - 3*xi1 - 2*xi2) / n_out,    # H2
        (n_in[2] + xi1 + xi2 - 2*xi4) / n_out,# CH3OH
        (n_in[3] + xi1 + xi4) / n_out,        # H2O
        (n_in[4] - xi2) / n_out,              # CO
        (n_in[5] + xi4) / n_out               # DME
    ])
    
    return x


def rxn_ext(xi, n_in, T, p):
    """System of equilibrium equations."""
    x = composition(xi, n_in)
    
    if np.any(np.isnan(x)) or np.any(x <= 0):
        return np.array([1e6, 1e6, 1e6])
    
    x_CO2, x_H2, x_CH3OH, x_H2O, x_CO, x_DME = x
    
    # Get equilibrium constants
    K1 = rxn_data_1(T)[-1]
    K2 = rxn_data_2(T)[-1]
    K4 = rxn_data_4(T)[-1]
    
    # Convert to K_x
    p_std = 1
    Kx1 = K1 * (p/p_std)**2
    Kx2 = K2 * (p/p_std)**2
    Kx4 = K4
    
    # Equilibrium equations
    res1 = Kx1 * x_CO2 * x_H2**3 - x_CH3OH * x_H2O
    res2 = Kx2 * x_CO * x_H2**2 - x_CH3OH
    res4 = Kx4 * x_CH3OH**2 - x_DME * x_H2O
    
    return [res1, res2, res4]


# Setup and solve
n_in = np.array([2.0, 6.0, 0.0, 0.0, 0.0, 0.0])  # CO2, H2, CH3OH, H2O, CO, DME
T_calc = np.linspace(100+273, 300+273, 40)
p_calc = np.array([20, 50, 100])

xi = np.empty([len(p_calc), len(T_calc), 3])

print("\n" + "=" * 60)
print("SOLVING CHEMICAL EQUILIBRIUM")
print("=" * 60)

for ip, p_val in enumerate(p_calc):
    print(f"\nPressure: {p_val} bar")
    
    for iT in range(len(T_calc)-1, -1, -1):  # High to low T
        T_val = T_calc[iT]
        
        xi_guess = np.array([0.3, -0.05, 0.05]) if iT == len(T_calc)-1 else xi[ip, iT+1, :]
        
        try:
            result = root(rxn_ext, xi_guess, args=(n_in, T_val, p_val), method='lm')
            
            if result.success:
                xi[ip, iT, :] = result.x
                if iT % 8 == 0:
                    print(f"  T={T_val-273:.0f}°C: ξ=[{result.x[0]:.4f}, {result.x[1]:.5f}, {result.x[2]:.4f}]")
            else:
                result = root(rxn_ext, xi_guess, args=(n_in, T_val, p_val), method='hybr')
                xi[ip, iT, :] = result.x if result.success else np.nan * np.ones(3)
        except Exception as e:
            print(f"  ERROR at T={T_val-273:.0f}°C: {e}")
            xi[ip, iT, :] = np.nan * np.ones(3)


#%%
# RESULTS: REACTION EXTENTS

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
    
    for ip, p_val in enumerate(p_calc):
        plt.plot(T_calc - 273, xi[ip, :, rxn_idx], colors[ip], 
                label=f"{p_val:.0f} bar", linewidth=2, marker='o', markersize=3)
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


#%%
# RESULTS: OUTLET COMPOSITION

# Calculate outlet flows
n_out_all = np.empty([len(p_calc), len(T_calc), 6])

for ip in range(len(p_calc)):
    for iT in range(len(T_calc)):
        if not np.isnan(xi[ip, iT, 0]):
            xi1, xi2, xi4 = xi[ip, iT, :]
            n_out_all[ip, iT, :] = [
                n_in[0] - xi1,                     # CO2
                n_in[1] - 3*xi1 - 2*xi2,           # H2
                n_in[2] + xi1 + xi2 - 2*xi4,       # CH3OH
                n_in[3] + xi1 + xi4,               # H2O
                n_in[4] - xi2,                     # CO
                n_in[5] + xi4                      # DME
            ]
        else:
            n_out_all[ip, iT, :] = np.nan * np.ones(6)

# Plot individual component flows
comp_names = ["CO₂", "H₂", "CH₃OH", "H₂O", "CO", "DME"]

for ic, comp_name in enumerate(comp_names):
    plt.figure(figsize=(6, 5))
    plt.grid()
    plt.title(f"Outlet flow: {comp_name}")
    plt.xlabel(r"$T\,/\,°C$")
    plt.ylabel(r"$\dot{n}_{\mathrm{out}}\,/\,\mathrm{mol\,s^{-1}}$")
    
    for ip, p_val in enumerate(p_calc):
        plt.plot(T_calc - 273, n_out_all[ip, :, ic], colors[ip],
                label=f"{p_val:.0f} bar", linewidth=2, marker='o', markersize=3)
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Plot all components at reference pressure
ip_ref = 1  # 50 bar
plt.figure(figsize=(8, 6))
plt.grid()
plt.title(f"Outlet composition at {p_calc[ip_ref]:.0f} bar")
plt.xlabel(r"$T\,/\,°C$")
plt.ylabel(r"$\dot{n}_{\mathrm{out}}\,/\,\mathrm{mol\,s^{-1}}$")

for ic, comp_name in enumerate(comp_names):
    plt.plot(T_calc - 273, n_out_all[ip_ref, :, ic],
            label=comp_name, linewidth=2, marker='o', markersize=3)

plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Calculate and plot molar fractions
x_out_all = np.empty([len(p_calc), len(T_calc), 6])

for ip in range(len(p_calc)):
    for iT in range(len(T_calc)):
        if not np.isnan(n_out_all[ip, iT, 0]):
            n_total = np.sum(n_out_all[ip, iT, :])
            x_out_all[ip, iT, :] = n_out_all[ip, iT, :] / n_total
        else:
            x_out_all[ip, iT, :] = np.nan * np.ones(6)

plt.figure(figsize=(8, 6))
plt.grid()
plt.title(f"Outlet molar fractions at {p_calc[ip_ref]:.0f} bar")
plt.xlabel(r"$T\,/\,°C$")
plt.ylabel(r"$x_{\mathrm{out}}$ / mol fraction")

for ic, comp_name in enumerate(comp_names):
    plt.plot(T_calc - 273, x_out_all[ip_ref, :, ic],
            label=comp_name, linewidth=2, marker='o', markersize=3)

plt.legend(loc='best')
plt.tight_layout()
plt.show()


#%%
# SUMMARY TABLE

T_idx = np.argmin(np.abs(T_calc - (250+273)))

print("\n" + "=" * 60)
print(f"OUTLET FLOWS SUMMARY AT T={T_calc[T_idx]-273:.0f}°C")
print("=" * 60)

for ip, p_val in enumerate(p_calc):
    print(f"\n{'─' * 60}")
    print(f"Pressure: {p_val:.0f} bar")
    print(f"{'─' * 60}")
    print(f"{'Component':<10} {'Inlet':>12} {'Outlet':>12} {'Change':>12} {'x_out':>10}")
    print("─" * 60)
    
    for ic, comp_name in enumerate(comp_names):
        n_out = n_out_all[ip, T_idx, ic]
        x_out = x_out_all[ip, T_idx, ic]
        delta_n = n_out - n_in[ic]
        print(f"{comp_name:<10} {n_in[ic]:>12.4f} {n_out:>12.4f} {delta_n:>12.4f} {x_out*100:>9.2f}%")
    
    print("─" * 60)
    print(f"{'Total':<10} {np.sum(n_in):>12.4f} {np.sum(n_out_all[ip, T_idx, :]):>12.4f}")
    
    # Performance metrics
    xi_vals = xi[ip, T_idx, :]
    conv_CO2 = (n_in[0] - n_out_all[ip, T_idx, 0]) / n_in[0] * 100
    sel_DME = n_out_all[ip, T_idx, 5] / (n_out_all[ip, T_idx, 2] + 2*n_out_all[ip, T_idx, 5]) * 100
    
    print(f"\nReaction extents: ξ₁={xi_vals[0]:.4f}, ξ₂={xi_vals[1]:.5f}, ξ₄={xi_vals[2]:.4f}")
    print(f"CO₂ conversion:   {conv_CO2:.2f}%")
    print(f"DME selectivity:  {sel_DME:.2f}% (C-basis)")

print("\n" + "=" * 60)