#%%
# Imports of packages
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

#%%

# create list of files in directory
os_list = os.listdir(path)
# provide length of list, which equals the number of experiments (nex)
nex = len(os_list)

# create empty list for concentrations, times and initials to store experimental results
exp_concs = []
t_eval = []
c_inits = []
for i in np.arange(0, nex):
    # iterate over all csv-files and append them, after transposing, as array to the list
    aux = pd.read_csv(path + r'\Exp_res_' + str(i) + '.csv')
    t_eval.append(aux.values.transpose()[0, :])         # time in s
    c_inits.append(aux.values.transpose()[1:, 0])       # initial concentration of component i in mol/m^3
    exp_concs.append(aux.values.transpose()[1:, :])     # timedependent concentration of component i in mol/m^3

# Create flattened array of results
nex = len(exp_concs)
exp_concs_flat = np.array([])
for i in np.arange(0, nex):
    exp_concs_flat = np.append(exp_concs_flat, exp_concs[i])

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

for exp in range(1, nex + 1):

    exp_idx = exp - 1

    fig, ax = plt.subplots(figsize=(10,10))
    cumulative = 0

    for comp in components:

        comp_idx = components.index(comp)

        ax.scatter(
            t_eval[exp_idx],
            exp_concs[exp_idx][comp_idx, :],
            label=f"Concentration {comp} Exp. {exp}",
            color=colors[comp],
            marker="o"
        )

        cumulative += exp_concs[exp_idx][comp_idx, :]

    # Plot cumulative concentration
    ax.scatter(
        t_eval[exp_idx],
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
# Components, that are not taking part in the reaction and their corresponding 
# concentration values in the dataframe are set to zero to avoid confusion and stability to the solution.

# remove negative concentrations
for i in range(nex):
    exp_concs[i] = np.clip(exp_concs[i], 0.0, None)

# binary initial conditions
for i in range(nex):
    c_inits[i] = np.where(c_inits[i] >= 0.5, 1.0, 0.0)

# manually zero inactive species
components = ["A", "B", "C", "D", "E", "F"]

comp_idx = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5
}

zero_dict = {
    2: ["A", "B", "E"],
    3: ["A", "B", "C", "E"],
    4: ["A", "B", "E"],
    5: ["B"],
    6: ["A", "B", "C", "E"]
}

for exp, comps in zero_dict.items():

    exp_idx = exp - 1

    for comp in comps:

        row = comp_idx[comp]

        exp_concs[exp_idx][row, :] = 0.0
        c_inits[exp_idx][row] = 0.0

# flatten data
exp_concs_flat = np.array([])

for i in range(nex):
    exp_concs_flat = np.append(exp_concs_flat, exp_concs[i])


def ode(t, c, k):
    """
    Calculate time derivatives of the ode for a batch STR
    Parameters
    ----------
    t: float
        time of the calculation
    c: array
        concentration values at time t, size [nc]
        c[0] = c_A; c[1] = c_B; c[2] = c_C; c[3] = c_D; c[4] = c_E
    k: array
        kinetic coefficients for all reactions, size [nr]
    Returns
    -------
    dcdt: array
        time derivatives of the concentration, size [nc]
    """
    dcdt = np.zeros_like(c)
    # calculating the rates for each reaction
    r0 = k[0] * c[0] * c[1]     # reaction 1: A + B -> C 
    r1 = k[1] * c[0] * c[2]     # reaction 2: A + C -> E 
    r2 = k[2] * c[2]            # reaction 3:     C -> D 
    r3 = k[3] * np.power(c[3], 2)            # reaction 4:     D -> F.

    # calculating the derivatives for each component
    dcdt[0] = - r0 - r1
    dcdt[1] = - r0
    dcdt[2] = + r0 - r1 - r2
    dcdt[3] = + r2 - r3
    dcdt[4] = + r1
    dcdt[5] = + r3
    return dcdt

def sim_exp(t, c_init, k):
    """
    Simulate single experiment using solve_ivp.
    Parameters
    ----------
    t: array
       array with the time points of the simulation, size [nt]
    c_init: array
        initial concentrations, size [nc]
    k: array
        kinetic coefficients for all reactions, size [nr]
    Returns
    -------
    c_sol: array
        concentration values at times defined by t, size [nc, nt]
    """
    t_sp = np.array([t[0], t[-1]])
    sol = integ.solve_ivp(fun=ode, t_span=t_sp, y0=c_init, method='LSODA', t_eval=t, args=[k])
    c_sol = sol.y
    return c_sol

def sim_multiple_exps(times, k0, k1, k2, k3, c_inits):
    """
    Simulate results for n experiments with nc components and 4 reactions. Each experiment has nt_i entries.
    Parameters
    ----------
    times: list
        List of arrays with times of sampling for each experiment, size [n][nt_i,]
    k0: float
        Kinetic parameter of first reaction
    k1: float
        Kinetic parameter of second reaction
    k2: float
        Kinetic parameter of third reaction
    k3: float
        Kinetic parameter of fourth reaction
    c_inits: list
        List of arrays with initial concentrations, size [n][nc]
    Returns
    -------
    sim concs: list
        List of arrays with calculated concentration values, size[n][nc, nt_i].
    """

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
    """
    Calculate the difference between simulation and experiment for each provided datum.
    Parameters
    ----------
    params: Parameters
        Parameter object containing all variables to be estimated
    times: list
        List of arrays with times of sampling for each experiment, size [n][nt_i,]
    c_inits: list
        List of arrays with initial concentrations, size [n][nc]
    data: list
        List of arrays with concentration values from experiment, size[n][nc, nt_i].
    Returns
    -------
    concs_flat - data: array
        flattened array of differences between simulation and experimental data, size[n*nc*nt_i]
    """
    # number of experiments from length
    sim_conc = sim_multiple_exps(
        times, 
        params["k0"], 
        params["k1"], 
        params["k2"], 
        params["k3"], 
        c_inits)

    nex = len(times)
    concs_flat = np.array([])
    for i in np.arange(0, nex):
        concs_flat = np.append(concs_flat, sim_conc[i])

    # return concs_flat - data

    # weighting
    sigma = 0.02 + 0.05 * np.abs(data)
    
    return (concs_flat - data) / sigma

# Parameters and lists for time, initial concentrations and concentrations

params = Parameters()
params.add('k0', value=0.2, min=0, max=10, vary=True)
params.add('k1', value=0.01, min=0, max=10, vary=True)
params.add('k2', value=0.05, min=0, max=10, vary=True)
params.add('k3', value=0.001, min=0, max=10, vary=True)

minner = Minimizer(residual, 
                   params, 
                   fcn_args=(t_eval, 
                             c_inits, 
                             exp_concs_flat))
result = minner.minimize()
report_fit(result)


model = Model(sim_multiple_exps, independent_vars=['times', 'c_inits'])
# initial_fit = model.eval(params, times=t_eval, c_inits=c_inits)
best_fit = sim_multiple_exps(times=t_eval, k0=result.params['k0'], k1=result.params['k1'], k2=result.params['k2'], k3=result.params['k3'], c_inits=c_inits)

for exp in range(1, nex + 1):

    exp_idx = exp - 1
    fig, ax = plt.subplots(figsize=(10,10))
    cumulative = 0
    cumulative_fit = 0
    fit_exp = best_fit[exp - 1]


    for comp in components:
        comp_idx = components.index(comp)
        ax.scatter(
            t_eval[exp_idx],
            exp_concs[exp_idx][comp_idx, :],
            label=f"Concentration {comp} Exp. {exp}",
            color=colors[comp],
            marker="o"
        )

        ax.plot(
            t_eval[exp_idx],
            fit_exp[comp_idx, :],
            label=f"Fit {comp} Exp. {exp}",
            color=colors[comp],
            linewidth=2,
            linestyle="--"
        )


        cumulative += exp_concs[exp_idx][comp_idx, :]
        cumulative_fit += fit_exp[comp_idx, :]

    # Plot cumulative concentration
    ax.scatter(
        t_eval[exp_idx],
        cumulative,
        label=f"Cumulative Concentration Exp. {exp}",
        color="cyan",
        marker="o"
    )

    ax.set_xlabel("t / s")
    ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
    ax.legend()
    ax.set_title(f"Experiment {exp}")

    # cumulative fitted curve
    ax.plot(
        t_eval[exp_idx],
        cumulative_fit,
        label=f"Cumulative Fit Exp. {exp}",
        color="cyan",
        linewidth=2
    )

    ax.set_xlabel("t / s")
    ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
    ax.set_title(f"Experiment {exp}")
    ax.legend()
    plt.show()

#%%
# New functions with varying reactionorder too

def ode_var_n(t, c, k, n):
    """
    Calculate time derivatives of the ode for a batch STR
    Parameters
    ----------
    t: float
        time of the calculation
    c: array
        concentration values at time t, size [nc]
        c[0] = c_A; c[1] = c_B; c[2] = c_C; c[3] = c_D; c[4] = c_E
    k: array
        kinetic coefficients for all reactions, size [nr]
    n: array
        reaction orders for all reactions, size [nr]
    Returns
    -------
    dcdt: array
        time derivatives of the concentration, size [nc]
    """
    dcdt = np.zeros_like(c)
    # calculating the rates for each reaction
    r0 = k[0] * np.power(c[0], n[0]) * np.power(c[1], n[1])     # reaction 1: A + B -> C 
    r1 = k[1] * np.power(c[0], n[2]) * np.power(c[2], n[3])     # reaction 2: A + C -> E 
    r2 = k[2] * np.power(c[2], n[4])            # reaction 3:     C -> D 
    r3 = k[3] * np.power(c[3], n[5])            # reaction 4:     D -> F

    # calculating the derivatives for each component
    dcdt[0] = - r0 - r1
    dcdt[1] = - r0
    dcdt[2] = + r0 - r1 - r2
    dcdt[3] = + r2 - r3
    dcdt[4] = + r1
    dcdt[5] = + r3
    return dcdt

def sim_exp_var_n(t, c_init, k, n):
    """
    Simulate single experiment using solve_ivp.
    Parameters
    ----------
    t: array
       array with the time points of the simulation, size [nt]
    c_init: array
        initial concentrations, size [nc]
    k: array
        kinetic coefficients for all reactions, size [nr]
    n: array
        reaction orders for all reactions, size [nr]
    Returns
    -------
    c_sol: array
        concentration values at times defined by t, size [nc, nt]
    """
    t_sp = np.array([t[0], t[-1]])
    sol = integ.solve_ivp(fun=ode_var_n, t_span=t_sp, y0=c_init, method='LSODA', t_eval=t, args=[k, n])
    c_sol = sol.y
    return c_sol

def sim_multiple_exps_var_n(times, k0, k1, k2, k3, n0, n1, n2, n3, n4, n5, c_inits):
    """
    Simulate results for n experiments with nc components and 4 reactions. Each experiment has nt_i entries.
    Parameters
    ----------
    times: list
        List of arrays with times of sampling for each experiment, size [n][nt_i,]
    k0: float
        Kinetic parameter of first reaction
    k1: float
        Kinetic parameter of second reaction
    k2: float
        Kinetic parameter of third reaction
    k3: float
        Kinetic parameter of fourth reaction
    n0: float
        Reaction order of first reaction with respect to Reaction 1 Component A
    n1: float
        Reaction order of second reaction with respect to Reaction 1 Component B
    n2: float
        Reaction order of third reaction with respect to Reaction 2 Component A
    n3: float
        Reaction order of fourth reaction with respect to Reaction 2 Component C
    n4: float
        Reaction order of fifth reaction with respect to Reaction 3 Component C
    n5: float
        Reaction order of sixth reaction with respect to Reaction 4 Component D
    c_inits: list
        List of arrays with initial concentrations, size [n][nc]
    Returns
    -------
    sim concs: list
        List of arrays with calculated concentration values, size[n][nc, nt_i].
    """

    sim_concs = []
    k = [k0, k1, k2, k3]
    n = [n0, n1, n2, n3, n4, n5]

    # iterate over all experiments
    nex = len(times)
    for i in np.arange(0, nex):
        # assign c and t to run simulation
        c_0 = c_inits[i]
        t = times[i]

        # run simulation for one experiment
        conc = sim_exp_var_n(t, c_0, k, n)
        sim_concs.append(conc)
    return sim_concs

def residual_var_n(params, times, c_inits, data):
    """
    Calculate the difference between simulation and experiment for each provided datum.
    Parameters
    ----------
    params: Parameters
        Parameter object containing all variables to be estimated
    times: list
        List of arrays with times of sampling for each experiment, size [n][nt_i,]
    c_inits: list
        List of arrays with initial concentrations, size [n][nc]
    data: list
        List of arrays with concentration values from experiment, size[n][nc, nt_i].
    Returns
    -------
    concs_flat - data: array
        flattened array of differences between simulation and experimental data, size[n*nc*nt_i]
    """
    # number of experiments from length
    sim_conc = sim_multiple_exps_var_n(
        times, 
        params["k0"], 
        params["k1"], 
        params["k2"], 
        params["k3"], 
        params["n0"],
        params["n1"],
        params["n2"],
        params["n3"],
        params["n4"],
        params["n5"],
        c_inits)

    nex = len(times)
    concs_flat = np.array([])
    for i in np.arange(0, nex):
        concs_flat = np.append(concs_flat, sim_conc[i])

    # return concs_flat - data

    # weighting
    sigma = 0.02 + 0.05 * np.abs(data)
    
    return (concs_flat - data) / sigma

#%%
# Parameters and lists for time, initial concentrations and concentrations

params = Parameters()
params.add('k0', value=0.2, min=0, max=10, vary=True)
params.add('k1', value=0.01, min=0, max=10, vary=True)
params.add('k2', value=0.05, min=0, max=10, vary=True)
params.add('k3', value=0.001, min=0, max=10, vary=True)
params.add('n0', value=1, min=0.8, max=2, vary=True)
params.add('n1', value=1, min=0.8, max=2, vary=True)
params.add('n2', value=1, min=0.8, max=2, vary=True)
params.add('n3', value=1, min=0.8, max=2, vary=True)
params.add('n4', value=1, min=0.8, max=2, vary=True)
params.add('n5', value=1, min=0.8, max=2, vary=True)

minner = Minimizer(residual_var_n, 
                   params, 
                   fcn_args=(t_eval, 
                             c_inits, 
                             exp_concs_flat))
result = minner.minimize()
report_fit(result)


model_var_n = Model(sim_multiple_exps_var_n, independent_vars=['times', 'c_inits'])
# initial_fit = model.eval(params, times=t_eval, c_inits=c_inits)
best_fit_var_n = sim_multiple_exps_var_n(times=t_eval, 
                                   k0=result.params['k0'], 
                                   k1=result.params['k1'], 
                                   k2=result.params['k2'], 
                                   k3=result.params['k3'], 
                                   n0=result.params['n0'], 
                                   n1=result.params['n1'], 
                                   n2=result.params['n2'], 
                                   n3=result.params['n3'], 
                                   n4=result.params['n4'], 
                                   n5=result.params['n5'], 
                                   c_inits=c_inits)

for exp in range(1, nex + 1):

    exp_idx = exp - 1
    fig, ax = plt.subplots(figsize=(10,10))
    cumulative = 0
    cumulative_fit = 0
    fit_exp_var_n = best_fit_var_n[exp - 1]


    for comp in components:
        comp_idx = components.index(comp)
        ax.scatter(
            t_eval[exp_idx],
            exp_concs[exp_idx][comp_idx, :],
            label=f"Concentration {comp} Exp. {exp}",
            color=colors[comp],
            marker="o"
        )

        ax.plot(
            t_eval[exp_idx],
            fit_exp_var_n[comp_idx, :],
            label=f"Fit {comp} Exp. {exp}",
            color=colors[comp],
            linewidth=2,
            linestyle="--"
        )


        cumulative += exp_concs[exp_idx][comp_idx, :]
        cumulative_fit_var_n += fit_exp_var_n[comp_idx, :]

    # Plot cumulative concentration
    ax.scatter(
        t_eval[exp_idx],
        cumulative,
        label=f"Cumulative Concentration Exp. {exp}",
        color="cyan",
        marker="o"
    )

    ax.set_xlabel("t / s")
    ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
    ax.legend()
    ax.set_title(f"Experiment {exp}")

    # cumulative fitted curve
    ax.plot(
        t_eval[exp_idx],
        cumulative_fit_var_n,
        label=f"Cumulative Fit Exp. {exp}",
        color="cyan",
        linewidth=2
    )

    ax.set_xlabel("t / s")
    ax.set_ylabel(r'$\mathrm{c}\; / \ \mathrm{\frac{mol}{m^3}}$')
    ax.set_title(f"Experiment {exp}")
    ax.legend()
    plt.show()
# %%
