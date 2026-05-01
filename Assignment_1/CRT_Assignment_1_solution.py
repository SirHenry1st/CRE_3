#%%
# Import

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import ICIW_Plots.colors as iciw_colors
plt.style.use("ICIWstyle")
import scipy.integrate as integ
from lmfit import Model, Minimizer, Parameters, report_fit

# Path to folder with csv-data
path = "2026_Task_data"

# Get all csv files in folder
files = sorted([f for f in os.listdir(path) if f.endswith(".csv")])

#%%
# Create Dataframe

# Empty dictionary for dataframe columns
data_dict = {}

# Loop through all experiments
for i, file in enumerate(files, start=1):
    
    # Read csv file
    aux = pd.read_csv(os.path.join(path, file), sep=",")
    
    # Rename and store columns
    for col in aux.columns:
        if col == "t":
            new_name = f"Exp_{i}_t"
        else:
            new_name = f"Exp_{i}_conc_{col}"
        
        data_dict[new_name] = aux[col].values

# Create final dataframe
df = pd.DataFrame(data_dict)

##%%
## First plots of all data
#
## Plotting first experiment
#fig, ax = plt.subplots(figsize=(10,10))
#ax.plot(df["Exp_1_t"], df["Exp_1_conc_A"], label = {"Concentration A Exp. 1"}, color=iciw_colors.CRIMSON, marker = "o")
#ax.plot(df["Exp_1_t"], df["Exp_1_conc_B"], label = {"Concentration B Exp. 1"}, color=iciw_colors.CERULEAN, marker = "o")
#ax.plot(df["Exp_1_t"], df["Exp_1_conc_C"], label = {"Concentration C Exp. 1"}, color=iciw_colors.KELLYGREEN, marker = "o")
#ax.plot(df["Exp_1_t"], df["Exp_1_conc_D"], label = {"Concentration D Exp. 1"}, color=iciw_colors.FLAME, marker = "o")
#ax.plot(df["Exp_1_t"], df["Exp_1_conc_E"], label = {"Concentration E Exp. 1"}, color=iciw_colors.DRAB, marker = "o")
#ax.plot(df["Exp_1_t"], df["Exp_1_conc_F"], label = {"Concentration F Exp. 1"}, color="magenta", marker = "o")
#ax.plot(df["Exp_1_t"], df["Exp_1_conc_A"]+df["Exp_1_conc_B"]+df["Exp_1_conc_C"]+df["Exp_1_conc_D"]+df["Exp_1_conc_E"]+df["Exp_1_conc_F"], label = {"Cumulative Concentration Exp. 1"}, color="cyan", marker = "o")
#ax.set_xlabel("t / s")
#ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
#ax.legend()
#plt.show()
#
## Plotting second experiment
#fig, ax = plt.subplots(figsize=(10,10))
#ax.plot(df["Exp_2_t"], df["Exp_2_conc_A"], label = {"Concentration A Exp. 2"}, color=iciw_colors.CRIMSON, marker = "o")
#ax.plot(df["Exp_2_t"], df["Exp_2_conc_B"], label = {"Concentration B Exp. 2"}, color=iciw_colors.CERULEAN, marker = "o")
#ax.plot(df["Exp_2_t"], df["Exp_2_conc_C"], label = {"Concentration C Exp. 2"}, color=iciw_colors.KELLYGREEN, marker = "o")
#ax.plot(df["Exp_2_t"], df["Exp_2_conc_D"], label = {"Concentration D Exp. 2"}, color=iciw_colors.FLAME, marker = "o")
#ax.plot(df["Exp_2_t"], df["Exp_2_conc_E"], label = {"Concentration E Exp. 2"}, color=iciw_colors.DRAB, marker = "o")
#ax.plot(df["Exp_2_t"], df["Exp_2_conc_F"], label = {"Concentration F Exp. 2"}, color="magenta", marker = "o")
#ax.plot(df["Exp_2_t"], df["Exp_2_conc_A"]+df["Exp_2_conc_B"]+df["Exp_2_conc_C"]+df["Exp_2_conc_D"]+df["Exp_2_conc_E"]+df["Exp_2_conc_F"], label = {"Cumulative Concentration Exp. 2"}, color="cyan", marker = "o")
#ax.set_xlabel("t / s")
#ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
#ax.legend()
#plt.show()
#
## Plotting third experiment
#fig, ax = plt.subplots(figsize=(10,10))
#ax.plot(df["Exp_3_t"], df["Exp_3_conc_A"], label = {"Concentration A Exp. 3"}, color=iciw_colors.CRIMSON, marker = "o")
#ax.plot(df["Exp_3_t"], df["Exp_3_conc_B"], label = {"Concentration B Exp. 3"}, color=iciw_colors.CERULEAN, marker = "o")
#ax.plot(df["Exp_3_t"], df["Exp_3_conc_C"], label = {"Concentration C Exp. 3"}, color=iciw_colors.KELLYGREEN, marker = "o")
#ax.plot(df["Exp_3_t"], df["Exp_3_conc_D"], label = {"Concentration D Exp. 3"}, color=iciw_colors.FLAME, marker = "o")
#ax.plot(df["Exp_3_t"], df["Exp_3_conc_E"], label = {"Concentration E Exp. 3"}, color=iciw_colors.DRAB, marker = "o")
#ax.plot(df["Exp_3_t"], df["Exp_3_conc_F"], label = {"Concentration F Exp. 3"}, color="magenta", marker = "o")
#ax.plot(df["Exp_3_t"], df["Exp_3_conc_A"]+df["Exp_3_conc_B"]+df["Exp_3_conc_C"]+df["Exp_3_conc_D"]+df["Exp_3_conc_E"]+df["Exp_3_conc_F"], label = {"Cumulative Concentration Exp. 3"}, color="cyan", marker = "o")
#ax.set_xlabel("t / s")
#ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
#ax.legend()
#plt.show()
#
## Plotting fourth experiment
#fig, ax = plt.subplots(figsize=(10,10))
#ax.plot(df["Exp_4_t"], df["Exp_4_conc_A"], label = {"Concentration A Exp. 4"}, color=iciw_colors.CRIMSON, marker = "o")
#ax.plot(df["Exp_4_t"], df["Exp_4_conc_B"], label = {"Concentration B Exp. 4"}, color=iciw_colors.CERULEAN, marker = "o")
#ax.plot(df["Exp_4_t"], df["Exp_4_conc_C"], label = {"Concentration C Exp. 4"}, color=iciw_colors.KELLYGREEN, marker = "o")
#ax.plot(df["Exp_4_t"], df["Exp_4_conc_D"], label = {"Concentration D Exp. 4"}, color=iciw_colors.FLAME, marker = "o")
#ax.plot(df["Exp_4_t"], df["Exp_4_conc_E"], label = {"Concentration E Exp. 4"}, color=iciw_colors.DRAB, marker = "o")
#ax.plot(df["Exp_4_t"], df["Exp_4_conc_F"], label = {"Concentration F Exp. 4"}, color="magenta", marker = "o")
#ax.plot(df["Exp_4_t"], df["Exp_4_conc_A"]+df["Exp_4_conc_B"]+df["Exp_4_conc_C"]+df["Exp_4_conc_D"]+df["Exp_4_conc_E"]+df["Exp_4_conc_F"], label = {"Cumulative Concentration Exp. 4"}, color="cyan", marker = "o")
#ax.set_xlabel("t / s")
#ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
#ax.legend()
#plt.show()
#
## Plotting fifth experiment
#fig, ax = plt.subplots(figsize=(10,10))
#ax.plot(df["Exp_5_t"], df["Exp_5_conc_A"], label = {"Concentration A Exp. 5"}, color=iciw_colors.CRIMSON, marker = "o")
#ax.plot(df["Exp_5_t"], df["Exp_5_conc_B"], label = {"Concentration B Exp. 5"}, color=iciw_colors.CERULEAN, marker = "o")
#ax.plot(df["Exp_5_t"], df["Exp_5_conc_C"], label = {"Concentration C Exp. 5"}, color=iciw_colors.KELLYGREEN, marker = "o")
#ax.plot(df["Exp_5_t"], df["Exp_5_conc_D"], label = {"Concentration D Exp. 5"}, color=iciw_colors.FLAME, marker = "o")
#ax.plot(df["Exp_5_t"], df["Exp_5_conc_E"], label = {"Concentration E Exp. 5"}, color=iciw_colors.DRAB, marker = "o")
#ax.plot(df["Exp_5_t"], df["Exp_5_conc_F"], label = {"Concentration F Exp. 5"}, color="magenta", marker = "o")
#ax.plot(df["Exp_5_t"], df["Exp_5_conc_A"]+df["Exp_5_conc_B"]+df["Exp_5_conc_C"]+df["Exp_5_conc_D"]+df["Exp_5_conc_E"]+df["Exp_5_conc_F"], label = {"Cumulative Concentration Exp. 5"}, color="cyan", marker = "o")
#ax.set_xlabel("t / s")
#ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
#ax.legend()
#plt.show()
#
## Plotting sixth experiment
#fig, ax = plt.subplots(figsize=(10,10))
#ax.plot(df["Exp_6_t"], df["Exp_6_conc_A"], label = {"Concentration A Exp. 6"}, color=iciw_colors.CRIMSON, marker = "o")
#ax.plot(df["Exp_6_t"], df["Exp_6_conc_B"], label = {"Concentration B Exp. 6"}, color=iciw_colors.CERULEAN, marker = "o")
#ax.plot(df["Exp_6_t"], df["Exp_6_conc_C"], label = {"Concentration C Exp. 6"}, color=iciw_colors.KELLYGREEN, marker = "o")
#ax.plot(df["Exp_6_t"], df["Exp_6_conc_D"], label = {"Concentration D Exp. 6"}, color=iciw_colors.FLAME, marker = "o")
#ax.plot(df["Exp_6_t"], df["Exp_6_conc_E"], label = {"Concentration E Exp. 6"}, color=iciw_colors.DRAB, marker = "o")
#ax.plot(df["Exp_6_t"], df["Exp_6_conc_F"], label = {"Concentration F Exp. 6"}, color="magenta", marker = "o")
#ax.plot(df["Exp_6_t"], df["Exp_6_conc_A"]+df["Exp_6_conc_B"]+df["Exp_6_conc_C"]+df["Exp_6_conc_D"]+df["Exp_6_conc_E"]+df["Exp_6_conc_F"], label = {"Cumulative Concentration Exp. 6"}, color="cyan", marker = "o")
#ax.set_xlabel("t / s")
#ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
#ax.legend()
#plt.show()

#%%

# Automatic plotting of all experiments

components = ["A", "B", "C", "D", "E", "F"]

colors = {
    "A": iciw_colors.CRIMSON,
    "B": iciw_colors.CERULEAN,
    "C": iciw_colors.KELLYGREEN,
    "D": iciw_colors.FLAME,
    "E": iciw_colors.DRAB,
    "F": "magenta"
}

# Number of experiments
n_exp = len([col for col in df.columns if col.endswith("_t")])

for exp in range(1, n_exp + 1):

    fig, ax = plt.subplots(figsize=(10,10))

    t_col = f"Exp_{exp}_t"
    cumulative = 0

    for comp in components:

        c_col = f"Exp_{exp}_conc_{comp}"

        # only plot if column exists
        if c_col in df.columns:
            ax.scatter(
                df[t_col],
                df[c_col],
                label=f"Concentration {comp} Exp. {exp}",
                color=colors[comp],
                marker="o"
            )

            cumulative += df[c_col]

    # Plot cumulative concentration
    ax.scatter(
        df[t_col],
        cumulative,
        label=f"Cumulative Concentration Exp. {exp}",
        color="cyan",
        marker="o"
    )

    ax.set_xlabel("t / s")
    ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
    ax.legend()
    ax.set_title(f"Experiment {exp}")

    plt.show()


# %%
# Reaction hypothesis

# A + B -> C
# C -> D
# A + C -> E
# D -> F
# all stoichiometry is 1 might be correct, but not sure about that
# 
# Components, that are not taking part in the reaction, are not shown in further the plots and corresponding 
# concentration values in the dataframe are set to zero to avoid confusion and stability to the solution.

cols_to_zero = [
    "Exp_2_conc_A",
    "Exp_2_conc_B",
    "Exp_2_conc_E",
    "Exp_3_conc_A",
    "Exp_3_conc_B",
    "Exp_3_conc_C",
    "Exp_3_conc_E",
    "Exp_4_conc_A",
    "Exp_4_conc_B",
    "Exp_4_conc_E",
    "Exp_5_conc_B",
    "Exp_6_conc_A",
    "Exp_6_conc_B",
    "Exp_6_conc_C",
    "Exp_6_conc_E"
]

# Create new dataframe without these columns
df_clean = df.copy()

df_clean[cols_to_zero] = 0.0

print(df_clean.head())

for exp in range(1, n_exp + 1):

    fig, ax = plt.subplots(figsize=(10,10))
    
    t_col = f"Exp_{exp}_t"
    cumulative = 0

    for comp in components:

        c_col = f"Exp_{exp}_conc_{comp}"

        # only plot if column exists
        if c_col in df_clean.columns:
            ax.scatter(
                df_clean[t_col],
                df_clean[c_col],
                label=f"Concentration {comp} Exp. {exp}",
                color=colors[comp],
                marker="o"
            )

            cumulative += df_clean[c_col]

    # Plot cumulative concentration
    ax.scatter(
        df_clean[t_col],
        cumulative,
        label=f"Cumulative Concentration Exp. {exp}",
        color="cyan",
        marker="o"
    )

    ax.set_xlabel("t / s")
    ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
    ax.legend()
    ax.set_title(f"Experiment {exp}")

    plt.show()

# %%
# Differential equations

def ode(t, c, k):
    
    dcdt = np.zeros_like(c)
    # calculating the rates
    r0 = k[0] * c[0] * c[1]     # A + B -> C
    r1 = k[1] * c[2]            # C -> D
    r2 = k[2] * c[0] * c[2]     # A + C -> E
    r3 = k[3] * c[3]            # D -> F

    # calculating the derivatives
    dcdt[0] = - r0 - r2         # A
    dcdt[1] = - r0              # B
    dcdt[2] = + r0 - r1 - r2    # C
    dcdt[3] = + r1 - r3         # D
    dcdt[4] = + r2              # E
    dcdt[5] = + r3              # F
    return dcdt


def sim_exp(t, c_init, k):
    t_sp = np.array([t[0], t[-1]])
    sol = integ.solve_ivp(fun=ode, t_span=t_sp, y0=c_init, method='LSODA', t_eval=t, args=[k])
    c_sol = sol.y
    return c_sol


def sim_multiple_exps(times, k0, k1, k2, k3, c_inits):
    sim_concs = []
    k = [k0, k1, k2, k3]

    # iterate over all experiments
    nex = len(times)
    for i in np.arange(0, nex):
        # assign c and t to run simulation
        c_0 = c_inits[i]
        t = times[i]

        # run simulation for one experiment
        conc = sim_exp(t, c_0, k)
        sim_concs.append(conc)
    return sim_concs


def residual(params, times, c_inits, data):
    # number of experiments from length
    sim_conc = sim_multiple_exps(times, params["k0"], params["k1"], params["k2"], params["k3"], c_inits)

    nex = len(times)
    concs_flat = np.array([])
    for i in np.arange(0, nex):
        concs_flat = np.append(concs_flat, sim_conc[i])

#%%

params = Parameters()
params.add('k0', value=0.5, min=0, max=10, vary=True)
params.add('k1', value=0.5, min=0, max=10, vary=True)
params.add('k2', value=0.5, min=0, max=10, vary=True)
params.add('k3', value=0.5, min=0, max=10, vary=True)

times   = []
c_inits = []
data    = []

for exp in range(1, n_exp + 1):
    t_col = f"Exp_{exp}_t"
    c_cols = [f"Exp_{exp}_conc_{comp}" for comp in components]

    if all(col in df_clean.columns for col in c_cols):
        times.append(df_clean[t_col].values)
        c_inits.append(df_clean[c_cols].iloc[0].values)
        data.append(df_clean[c_cols].values.flatten())

print(times)
print(c_inits)
print(data)

# #%%
# 
# times_2 = []
# c_inits_2 = []
# data_2 = []
# 
# for exp in range(1, n_exp + 1):
# 
#     # time column
#     t_col = f"Exp_{exp}_t"
#     t = df_clean[t_col].values
#     times_2.append(t)
# 
#     # initial concentrations
#     c0 = []
# 
#     # full concentration matrix
#     conc_matrix = []
# 
#     for comp in components:
# 
#         c_col = f"Exp_{exp}_conc_{comp}"
# 
#         c_vals = df_clean[c_col].values
# 
#         c0.append(c_vals[0])          # first value = initial value
#         conc_matrix.append(c_vals)    # full time profile
# 
#     c_inits_2.append(np.array(c0))
#     data_2.append(np.array(conc_matrix))
# 
# print(times_2)
# print(c_inits_2)
# print(data_2)
# 
#%%

minner = Minimizer(residual, params, fcn_args=(times, c_inits, data))
result = minner.minimize()
report_fit(result)