#%%
# Import
import numpy as np
import scipy.integrate as integ 
from scipy.integrate import solve_ivp, solve_bvp 
import matplotlib.pyplot as plt 
import pandas as pd

#%%
# Solver validation

def conversion(c_A, c_A_in):
    """
    Calculate conversion of component A.
    
    Parameters
    ----------
    c_A : float or array
        Local concentration of component A.
    c_A_in : float
        Inlet concentration of component A.
        
    Returns
    -------
    X_A : float or array
        Conversion of component A.
    """
    return (c_A_in - c_A) / c_A_in


# Given reactor and feed data
L = 10.0                       # reactor length, m
d_R = 32e-3                    # reactor diameter, m
d_up = 150e-3                  # upstream pipe diameter, m
V_dot = 1.0e-3                 # volumetric flow rate, m3/s, from 1 L/s
c_A_in = 100.0                 # inlet concentration of A, mol/m3
c_B_in = 0.0                   # inlet concentration of B, mol/m3

# First implementation: first-order kinetics
# This value is selected as a reference value because the task statement does not provide k.
k = 0.2                        # rate constant, s^-1
reaction_order = 1.0

# Cross-sectional areas
A_R = np.pi * d_R**2 / 4
A_up = np.pi * d_up**2 / 4

# Superficial velocities
u = V_dot / A_R
u_up = V_dot / A_up

# Mean residence time in the reactor
tau = L / u

def pfr_ivp_rhs(z, y, k, u):
    """
    Right-hand side of the ideal plug-flow reactor model.

    y[0] = c_A
    y[1] = c_B
    """
    c_A, c_B = y

    r = k * c_A

    dc_A_dz = -r / u
    dc_B_dz = r / u

    return [dc_A_dz, dc_B_dz]


# Axial grid for evaluation
z_eval = np.linspace(0, L, 500)

# Solve ideal PFR model
ivp_solution = solve_ivp(
    fun=lambda z, y: pfr_ivp_rhs(z, y, k, u),
    t_span=(0, L),
    y0=[c_A_in, c_B_in],
    t_eval=z_eval,
    method="RK45",
    rtol=1e-9,
    atol=1e-11
)

if not ivp_solution.success:
    raise RuntimeError(ivp_solution.message)

# Store results
pfr_df = pd.DataFrame({
    "z / m": ivp_solution.t,
    "c_A / mol m^-3": ivp_solution.y[0],
    "c_B / mol m^-3": ivp_solution.y[1],
})

pfr_df["X_A / -"] = conversion(pfr_df["c_A / mol m^-3"], c_A_in)

# Analytical solution
pfr_df["c_A analytical / mol m^-3"] = c_A_in * np.exp(-k * pfr_df["z / m"] / u)
pfr_df["X_A analytical / -"] = conversion(pfr_df["c_A analytical / mol m^-3"], c_A_in)

# Error between numerical and analytical solution
max_error_cA = np.max(
    np.abs(pfr_df["c_A / mol m^-3"] - pfr_df["c_A analytical / mol m^-3"])
)

max_error_X = np.max(
    np.abs(pfr_df["X_A / -"] - pfr_df["X_A analytical / -"])
)

# Summary table
pfr_summary = pd.DataFrame([
    {
        "Model": "Ideal PFR IVP",
        "Residence time / s": tau,
        "c_A,out / mol m^-3": pfr_df["c_A / mol m^-3"].iloc[-1],
        "c_B,out / mol m^-3": pfr_df["c_B / mol m^-3"].iloc[-1],
        "X_A,out / -": pfr_df["X_A / -"].iloc[-1],
        "Max. error c_A / mol m^-3": max_error_cA,
        "Max. error X_A / -": max_error_X,
    }
])


# Concentration profiles
plt.figure()
plt.plot(
    pfr_df["z / m"],
    pfr_df["c_A / mol m^-3"],
    label="c_A numerical"
)
plt.plot(
    pfr_df["z / m"],
    pfr_df["c_B / mol m^-3"],
    label="c_B numerical"
)
plt.xlabel("Reactor length z / m")
plt.ylabel("Concentration / mol m^-3")
plt.title("Ideal PFR concentration profiles")
plt.legend()
plt.tight_layout()
plt.show()

# Numerical vs analytical validation
plt.figure()
plt.plot(
    pfr_df["z / m"],
    pfr_df["c_A / mol m^-3"],
    label="Numerical IVP"
)
plt.plot(
    pfr_df["z / m"],
    pfr_df["c_A analytical / mol m^-3"],
    "--",
    label="Analytical solution"
)
plt.xlabel("Reactor length z / m")
plt.ylabel("Concentration of A / mol m^-3")
plt.title("Validation of ideal PFR numerical solution")
plt.legend()
plt.tight_layout()
plt.show()

# Conversion profile
plt.figure()
plt.plot(
    pfr_df["z / m"],
    pfr_df["X_A / -"],
    label="Conversion of A"
)
plt.xlabel("Reactor length z / m")
plt.ylabel("Conversion X_A / -")
plt.title("Ideal PFR conversion profile")
plt.legend()
plt.tight_layout()
plt.show()


# %%
# Boundary value problem

def dispersion_bvp_rhs(z, y, k, u, D_ax):
    """
    Right-hand side of the axial-dispersion model, first-order system.

    y[0] = c_A
    y[1] = dc_A/dz
    y[2] = c_B
    y[3] = dc_B/dz
    """
    c_A, dcA_dz, c_B, dcB_dz = y

    # from  u*dcA/dz = D_ax*d2cA/dz2 - k*c_A
    d2cA_dz2 = (u * dcA_dz + k * c_A) / D_ax

    # from  u*dcB/dz = D_ax*d2cB/dz2 + k*c_A
    d2cB_dz2 = (u * dcB_dz - k * c_A) / D_ax

    return np.vstack((dcA_dz, d2cA_dz2, dcB_dz, d2cB_dz2))


def dispersion_bc(ya, yb, c_A_in, c_B_in, u, D_ax):
    """
    Danckwerts boundary conditions at z=0 and z=L for both species.
    """
    cA0, dcA0, cB0, dcB0 = ya   # state at z=0
    cAL, dcAL, cBL, dcBL = yb   # state at z=L

    # z = 0: Danckwerts inlet condition
    res_A0 = cA0 - (D_ax / u) * dcA0 - c_A_in
    res_B0 = cB0 - (D_ax / u) * dcB0 - c_B_in

    # z = L: zero-gradient outlet condition
    res_AL = dcAL
    res_BL = dcBL

    return np.array([res_A0, res_AL, res_B0, res_BL])


# Choose a high Bo for the validation run (weak dispersion -> should match IVP)
Bo_high = 1e4
D_ax = u * L / Bo_high

# Mesh and initial guess
z_mesh = np.linspace(0, L, 200)
y_guess = np.zeros((4, z_mesh.size))
y_guess[0] = c_A_in   # rough guess: c_A starts near inlet value
y_guess[2] = 0.0      # rough guess: c_B starts at 0

bvp_solution = solve_bvp(
    lambda z, y: dispersion_bvp_rhs(z, y, k, u, D_ax),
    lambda ya, yb: dispersion_bc(ya, yb, c_A_in, c_B_in, u, D_ax),
    z_mesh,
    y_guess
)

if not bvp_solution.success:
    raise RuntimeError(bvp_solution.message)

z_plot = np.linspace(0, L, 500)
y_plot = bvp_solution.sol(z_plot)
c_A_bvp = y_plot[0]
c_B_bvp = y_plot[2]

# Comparison: axial-dispersion BVP (high Bo) vs ideal PFR IVP
plt.figure()
plt.plot(
    pfr_df["z / m"],
    pfr_df["c_A / mol m^-3"],
    label="c_A ideal PFR (IVP)"
)
plt.plot(
    z_plot,
    c_A_bvp,
    "--",
    label=f"c_A dispersion model, Bo={Bo_high:.0f} (BVP)"
)
plt.xlabel("Reactor length z / m")
plt.ylabel("Concentration of A / mol m^-3")
plt.title("Validation: dispersion model (high Bo) vs. ideal PFR")
plt.legend()
plt.tight_layout()
plt.show()

# Same comparison for component B
plt.figure()
plt.plot(
    pfr_df["z / m"],
    pfr_df["c_B / mol m^-3"],
    label="c_B ideal PFR (IVP)"
)
plt.plot(
    z_plot,
    c_B_bvp,
    "--",
    label=f"c_B dispersion model, Bo={Bo_high:.0f} (BVP)"
)
plt.xlabel("Reactor length z / m")
plt.ylabel("Concentration of B / mol m^-3")
plt.title("Validation: dispersion model (high Bo) vs. ideal PFR")
plt.legend()
plt.tight_layout()
plt.show()

# Conversion comparison + quantitative error at outlet
X_A_bvp = conversion(c_A_bvp, c_A_in)

plt.figure()
plt.plot(
    pfr_df["z / m"],
    pfr_df["X_A / -"],
    label="X_A ideal PFR (IVP)"
)
plt.plot(
    z_plot,
    X_A_bvp,
    "--",
    label=f"X_A dispersion model, Bo={Bo_high:.0f} (BVP)"
)
plt.xlabel("Reactor length z / m")
plt.ylabel("Conversion X_A / -")
plt.title("Validation: conversion profile, dispersion model vs. ideal PFR")
plt.legend()
plt.tight_layout()
plt.show()

# Quantitative check at the outlet
X_A_out_ivp = pfr_df["X_A / -"].iloc[-1]
X_A_out_bvp = X_A_bvp[-1]
rel_diff = abs(X_A_out_bvp - X_A_out_ivp) / X_A_out_ivp

validation_summary = pd.DataFrame([{
    "Model": "Ideal PFR (IVP)",
    "Bo": np.inf,
    "X_A,out / -": X_A_out_ivp,
}, {
    "Model": "Dispersion model (BVP)",
    "Bo": Bo_high,
    "X_A,out / -": X_A_out_bvp,
}])
print(f"Relative difference in outlet conversion: {rel_diff:.2%}")