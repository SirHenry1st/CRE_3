#%%
# Import

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import ICIW_Plots.colors as iciw_colors

plt.style.use("ICIWstyle")

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

#%%
# First plots of all data

# Plotting first experiment
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(df["Exp_1_t"], df["Exp_1_conc_A"], label = {"Concentration A Exp. 1"}, color=iciw_colors.CRIMSON, marker = "o")
ax.plot(df["Exp_1_t"], df["Exp_1_conc_B"], label = {"Concentration B Exp. 1"}, color=iciw_colors.CERULEAN, marker = "o")
ax.plot(df["Exp_1_t"], df["Exp_1_conc_C"], label = {"Concentration C Exp. 1"}, color=iciw_colors.KELLYGREEN, marker = "o")
ax.plot(df["Exp_1_t"], df["Exp_1_conc_D"], label = {"Concentration D Exp. 1"}, color=iciw_colors.FLAME, marker = "o")
ax.plot(df["Exp_1_t"], df["Exp_1_conc_E"], label = {"Concentration E Exp. 1"}, color=iciw_colors.DRAB, marker = "o")
ax.plot(df["Exp_1_t"], df["Exp_1_conc_F"], label = {"Concentration F Exp. 1"}, color="magenta", marker = "o")
ax.plot(df["Exp_1_t"], df["Exp_1_conc_A"]+df["Exp_1_conc_B"]+df["Exp_1_conc_C"]+df["Exp_1_conc_D"]+df["Exp_1_conc_E"]+df["Exp_1_conc_F"], label = {"Cumulative Concentration Exp. 1"}, color="cyan", marker = "o")
ax.set_xlabel("t / s")
ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
ax.legend()
plt.show()

# Plotting second experiment
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(df["Exp_2_t"], df["Exp_2_conc_A"], label = {"Concentration A Exp. 2"}, color=iciw_colors.CRIMSON, marker = "o")
ax.plot(df["Exp_2_t"], df["Exp_2_conc_B"], label = {"Concentration B Exp. 2"}, color=iciw_colors.CERULEAN, marker = "o")
ax.plot(df["Exp_2_t"], df["Exp_2_conc_C"], label = {"Concentration C Exp. 2"}, color=iciw_colors.KELLYGREEN, marker = "o")
ax.plot(df["Exp_2_t"], df["Exp_2_conc_D"], label = {"Concentration D Exp. 2"}, color=iciw_colors.FLAME, marker = "o")
ax.plot(df["Exp_2_t"], df["Exp_2_conc_E"], label = {"Concentration E Exp. 2"}, color=iciw_colors.DRAB, marker = "o")
ax.plot(df["Exp_2_t"], df["Exp_2_conc_F"], label = {"Concentration F Exp. 2"}, color="magenta", marker = "o")
ax.plot(df["Exp_2_t"], df["Exp_2_conc_A"]+df["Exp_2_conc_B"]+df["Exp_2_conc_C"]+df["Exp_2_conc_D"]+df["Exp_2_conc_E"]+df["Exp_2_conc_F"], label = {"Cumulative Concentration Exp. 2"}, color="cyan", marker = "o")
ax.set_xlabel("t / s")
ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
ax.legend()
plt.show()

# Plotting third experiment
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(df["Exp_3_t"], df["Exp_3_conc_A"], label = {"Concentration A Exp. 3"}, color=iciw_colors.CRIMSON, marker = "o")
ax.plot(df["Exp_3_t"], df["Exp_3_conc_B"], label = {"Concentration B Exp. 3"}, color=iciw_colors.CERULEAN, marker = "o")
ax.plot(df["Exp_3_t"], df["Exp_3_conc_C"], label = {"Concentration C Exp. 3"}, color=iciw_colors.KELLYGREEN, marker = "o")
ax.plot(df["Exp_3_t"], df["Exp_3_conc_D"], label = {"Concentration D Exp. 3"}, color=iciw_colors.FLAME, marker = "o")
ax.plot(df["Exp_3_t"], df["Exp_3_conc_E"], label = {"Concentration E Exp. 3"}, color=iciw_colors.DRAB, marker = "o")
ax.plot(df["Exp_3_t"], df["Exp_3_conc_F"], label = {"Concentration F Exp. 3"}, color="magenta", marker = "o")
ax.plot(df["Exp_3_t"], df["Exp_3_conc_A"]+df["Exp_3_conc_B"]+df["Exp_3_conc_C"]+df["Exp_3_conc_D"]+df["Exp_3_conc_E"]+df["Exp_3_conc_F"], label = {"Cumulative Concentration Exp. 3"}, color="cyan", marker = "o")
ax.set_xlabel("t / s")
ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
ax.legend()
plt.show()

# Plotting fourth experiment
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(df["Exp_4_t"], df["Exp_4_conc_A"], label = {"Concentration A Exp. 4"}, color=iciw_colors.CRIMSON, marker = "o")
ax.plot(df["Exp_4_t"], df["Exp_4_conc_B"], label = {"Concentration B Exp. 4"}, color=iciw_colors.CERULEAN, marker = "o")
ax.plot(df["Exp_4_t"], df["Exp_4_conc_C"], label = {"Concentration C Exp. 4"}, color=iciw_colors.KELLYGREEN, marker = "o")
ax.plot(df["Exp_4_t"], df["Exp_4_conc_D"], label = {"Concentration D Exp. 4"}, color=iciw_colors.FLAME, marker = "o")
ax.plot(df["Exp_4_t"], df["Exp_4_conc_E"], label = {"Concentration E Exp. 4"}, color=iciw_colors.DRAB, marker = "o")
ax.plot(df["Exp_4_t"], df["Exp_4_conc_F"], label = {"Concentration F Exp. 4"}, color="magenta", marker = "o")
ax.plot(df["Exp_4_t"], df["Exp_4_conc_A"]+df["Exp_4_conc_B"]+df["Exp_4_conc_C"]+df["Exp_4_conc_D"]+df["Exp_4_conc_E"]+df["Exp_4_conc_F"], label = {"Cumulative Concentration Exp. 4"}, color="cyan", marker = "o")
ax.set_xlabel("t / s")
ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
ax.legend()
plt.show()

# Plotting fifth experiment
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(df["Exp_5_t"], df["Exp_5_conc_A"], label = {"Concentration A Exp. 5"}, color=iciw_colors.CRIMSON, marker = "o")
ax.plot(df["Exp_5_t"], df["Exp_5_conc_B"], label = {"Concentration B Exp. 5"}, color=iciw_colors.CERULEAN, marker = "o")
ax.plot(df["Exp_5_t"], df["Exp_5_conc_C"], label = {"Concentration C Exp. 5"}, color=iciw_colors.KELLYGREEN, marker = "o")
ax.plot(df["Exp_5_t"], df["Exp_5_conc_D"], label = {"Concentration D Exp. 5"}, color=iciw_colors.FLAME, marker = "o")
ax.plot(df["Exp_5_t"], df["Exp_5_conc_E"], label = {"Concentration E Exp. 5"}, color=iciw_colors.DRAB, marker = "o")
ax.plot(df["Exp_5_t"], df["Exp_5_conc_F"], label = {"Concentration F Exp. 5"}, color="magenta", marker = "o")
ax.plot(df["Exp_5_t"], df["Exp_5_conc_A"]+df["Exp_5_conc_B"]+df["Exp_5_conc_C"]+df["Exp_5_conc_D"]+df["Exp_5_conc_E"]+df["Exp_5_conc_F"], label = {"Cumulative Concentration Exp. 5"}, color="cyan", marker = "o")
ax.set_xlabel("t / s")
ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
ax.legend()
plt.show()

# Plotting sixth experiment
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(df["Exp_6_t"], df["Exp_6_conc_A"], label = {"Concentration A Exp. 6"}, color=iciw_colors.CRIMSON, marker = "o")
ax.plot(df["Exp_6_t"], df["Exp_6_conc_B"], label = {"Concentration B Exp. 6"}, color=iciw_colors.CERULEAN, marker = "o")
ax.plot(df["Exp_6_t"], df["Exp_6_conc_C"], label = {"Concentration C Exp. 6"}, color=iciw_colors.KELLYGREEN, marker = "o")
ax.plot(df["Exp_6_t"], df["Exp_6_conc_D"], label = {"Concentration D Exp. 6"}, color=iciw_colors.FLAME, marker = "o")
ax.plot(df["Exp_6_t"], df["Exp_6_conc_E"], label = {"Concentration E Exp. 6"}, color=iciw_colors.DRAB, marker = "o")
ax.plot(df["Exp_6_t"], df["Exp_6_conc_F"], label = {"Concentration F Exp. 6"}, color="magenta", marker = "o")
ax.plot(df["Exp_6_t"], df["Exp_6_conc_A"]+df["Exp_6_conc_B"]+df["Exp_6_conc_C"]+df["Exp_6_conc_D"]+df["Exp_6_conc_E"]+df["Exp_6_conc_F"], label = {"Cumulative Concentration Exp. 6"}, color="cyan", marker = "o")
ax.set_xlabel("t / s")
ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
ax.legend()
plt.show()

# %%
# Reaction hypothesis

# A + B -> C
# C -> D
# A + C -> E
# D -> F
# all stoichiometry is 1 might be correct, but not sure about that

