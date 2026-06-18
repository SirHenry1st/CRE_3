#%%
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import solve_ivp

#%%

E_A = 72500      # activation energy J/mol
k_0 = 5.01E06       # pre-exponential factor 1/s
R = 8.314        # gas constant J/(mol K)
H_R = -92200     # reaction enthalpy J/mol

d_R = 0.03       # tube diameter m
L_R = 300        # reactor length m
h_wall = 500     # wall heat transfer coefficient W/(m^2 K)  [= 0.5 kJ/(s m^2 K)]

a_wall = 4 / d_R           # specific wall area m^2_wall/m^3_reactor  (= pi*d / (pi*d^2/4))

def kinetics(T, c_1):
    """Arrhenius reaction rate, first order in c_1."""
    k_1 = k_0 * np.exp(-E_A / (R * T))   # rate constant 1/s
    r = k_1 * c_1                          # reaction rate mol/(m^3 s)
    return r

#%%

M_WA = 18.01 # Molar mass of water in g/mol
M_EG = 62.07 # Molar mass of ethylene glycol in g/mol
M_EO = 44.05 # Molar mass of ethylene oxide in g/mol

R_WA = 8.314 / M_WA # Specific heat capacity of water in 
R_EG = 8.314 / M_EG # Specific heat capacity of ethylene glycol in 
R_EO = 8.314 / M_EO # Specific heat capacity of ethylene oxide in 

T_C_EG = 724.05 # Critical temperature of ethylene glycol in K
T_C_WA = 647.10 # Critical temperature of water in K 
T_C_EO = 469.15 # Critical temperature of ethylene oxide in K 

c_p_EG_params = np.array([0, 33.1585, -25.9580, 0, 0, 0]) # Parameters for the specific heat capacity of ethylene glycol as a function of temperature
c_p_WA_params = np.array([0.2399, 12.8647, -33.6392, 104.7686, -155.4709, 92.3726]) # Parameters for the specific heat capacity of water as a function of temperature

# Heat capacity of ethylene glycol dependent on temperature in J/kg-K
def c_p_EG(T):
    return R_EG * (c_p_EG_params[0] / (1- (T / T_C_EG)) + c_p_EG_params[1] + c_p_EG_params[2] * (1 - (T / T_C_EG)) + c_p_EG_params[3] * (1 - (T / T_C_EG))**2 + c_p_EG_params[4] * (1 - (T / T_C_EG))**3 + c_p_EG_params[5] * (1 - (T / T_C_EG))**4) * 1000

# Heat capacity of water dependent on temperature in J/kg-K
def c_p_WA(T):
    return R_WA * (c_p_WA_params[0] / (1- (T / T_C_WA)) + c_p_WA_params[1] + c_p_WA_params[2] * (1 - (T / T_C_WA)) + c_p_WA_params[3] * (1 - (T / T_C_WA))**2 + c_p_WA_params[4] * (1 - (T / T_C_WA))**3 + c_p_WA_params[5] * (1 - (T / T_C_WA))**4) * 1000

c_p_EO_values = np.array([1.88, 1.91, 1.95, 2.00, 2.06, 2.15, 2.27, 2.40]) * 1000 # Specific heat capacity values for ethylene oxide in J/kg-K at the corresponding temperature values in
T_EO_values = np.array([-40, -20, 0, 20, 40, 60, 80, 100]) + 273.15 # Temperature values for the specific heat capacity of ethylene oxide in K

# Interpolate the specific heat capacity of ethylene oxide as a function of temperature
c_p_EO_interp = interp1d(T_EO_values, c_p_EO_values, kind='linear', fill_value=(c_p_EO_values[0], c_p_EO_values[-1]), bounds_error=False) 


# Ideal mixture of specific heat capacities in J kg-1 K-1
def c_p_mix(T, x_arr):
    x_WA = x_arr[0]
    x_EG = x_arr[1]
    x_EO = x_arr[2]

    w_EG = w(x_arr, x_EG, M_EG)
    w_WA = w(x_arr, x_WA, M_WA)
    w_EO = w(x_arr, x_EO, M_EO)

    return w_EG * c_p_EG(T) + w_WA * c_p_WA(T) + w_EO * c_p_EO_interp(T)

rho_EG_params = np.array([1305.5931, -1374.2561, 1691.0501, -665.0358]) # Parameters for the density of ethylene glycol as a function of temperature
rho_WA_params = np.array([1094.0233, -1813.2295, 3863.9557, -2479.8130]) # Parameters for the density of water as a function of temperature
rho_EO_params = np.array([757.9994, -286.5638, 583.1649, -177.0206]) # Parameters for the density of ethylene oxide as a function of temperature (not used in this code)

rho_EG_c = 325 # Critical density of ethylene glycol in kg/m^3
rho_WA_c = 322 # Critical density of water in kg/m^3
rho_EO_c = 314 # Critical density of ethylene oxide in kg/m^3

# Density of water dependent on temperature
def rho_WA(T):
    return rho_WA_c + rho_WA_params[0] * (1 - (T / T_C_WA)) ** 0.35 + rho_WA_params[1] * (1 - (T / T_C_WA))**(2/3) + rho_WA_params[2] * (1 - (T / T_C_WA)) + rho_WA_params[3] * (1 - (T / T_C_WA))**(4/3)

# Density of ethylene glycol dependent on temperature
def rho_EG(T):
    return rho_EG_c + rho_EG_params[0] * (1 - (T / T_C_EG)) ** 0.35 + rho_EG_params[1] * (1 - (T / T_C_EG))**(2/3) + rho_EG_params[2] * (1 - (T / T_C_EG)) + rho_EG_params[3] * (1 - (T / T_C_EG))**(4/3)

def rho_EO(T):
    return rho_EO_c + rho_EO_params[0] * (1 - (T / T_C_EO)) ** 0.35 + rho_EO_params[1] * (1 - (T / T_C_EO))**(2/3) + rho_EO_params[2] * (1 - (T / T_C_EO)) + rho_EO_params[3] * (1 - (T / T_C_EO))**(4/3)

print(rho_EG(273.15 + 20))
print(rho_WA(273.15 + 20))
print(rho_EO(273.15 + 20))

# Mass fraction of component x in the mixture
def w(x_arr, x, M):
    x_WA = x_arr[0]
    x_EG = x_arr[1]
    x_EO = x_arr[2]
    return x * M / (x_EG * M_EG + x_WA * M_WA + x_EO * M_EO)

# Density of the mixture dependent on temperature and composition
def rho_mix(T, x_arr):
    x_WA = x_arr[0]
    x_EG = x_arr[1]
    x_EO = x_arr[2]

    w_EG = w(x_arr, x_EG, M_EG)
    w_WA = w(x_arr, x_WA, M_WA)
    w_EO = w(x_arr, x_EO, M_EO)

    return 1 / (w_EG / rho_EG(T) + w_WA / rho_WA(T) + w_EO / rho_EO(T))

def x_i(c_arr, c_i):
    return c_i / np.sum(c_arr)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(30, 12))
T_range = np.arange(20, 150, 1) + 273.15
ax[0, 0].plot(T_range - 273.15, c_p_EG(T_range), label='Ethylene Glycol')
ax[0, 0].plot(T_range - 273.15, c_p_WA(T_range), label='Water')
ax[0, 0].plot(T_range - 273.15, c_p_EO_interp(T_range), label='Ethylene Oxide')
ax[0, 0].set_xlabel('Temperature (°C)')
ax[0, 0].set_ylabel('Specific Heat Capacity (J/kg-K)')
ax[0, 0].set_title('Specific Heat Capacity vs. Temperature')
ax[0, 0].legend()

ax[0, 1].plot(T_range - 273.15, rho_EG(T_range), label='Ethylene Glycol')
ax[0, 1].plot(T_range - 273.15, rho_WA(T_range), label='Water')
ax[0, 1].plot(T_range - 273.15, rho_EO(T_range), label='Ethylene Oxide')
ax[0, 1].set_xlabel('Temperature (°C)')
ax[0, 1].set_ylabel('Density (kg/m^3)')
ax[0, 1].set_title('Density vs. Temperature')
ax[0, 1].legend()

ax[1, 0].plot(T_range - 273.15, c_p_mix(T_range,  np.array([0.5, 0.5, 0])), label='50 mol% EG, 50 mol% WA')
ax[1, 0].plot(T_range - 273.15, c_p_mix(T_range, np.array([0.5, 0, 0.5])), label='50 mol% EG, 50 mol% EO')
ax[1, 0].plot(T_range - 273.15, c_p_mix(T_range, np.array([0, 0.5, 0.5])), label='50 mol% EO, 50 mol% WA')
ax[1, 0].set_xlabel('Temperature (°C)')
ax[1, 0].set_ylabel('Specific Heat Capacity (J/kg-K)')
ax[1, 0].set_title('Specific Heat Capacity of Mixture vs. Temperature')
ax[1, 0].legend() 

ax[1, 1].plot(T_range - 273.15, rho_mix(T_range, np.array([0.5, 0.5, 0])), label='50 mol% EG, 50 mol% WA')
ax[1, 1].plot(T_range - 273.15, rho_mix(T_range, np.array([0.5, 0, 0.5])), label='50 mol% EG, 50 mol% EO')
ax[1, 1].plot(T_range - 273.15, rho_mix(T_range, np.array([0, 0.5, 0.5])), label='50 mol% EO, 50 mol% WA')
ax[1, 1].set_xlabel('Temperature (°C)')
ax[1, 1].set_ylabel('Density (kg/m^3)')
ax[1, 1].set_title('Density of Mixture vs. Temperature')
ax[1, 1].legend()
plt.show()
#%%


#%%



def A(d):
    return np.pi * (d / 2) ** 2

def n_dot(u, rho, d, M):
    return u * rho * A(d) / (M * 0.001)

def u(n_dot, rho, d, M):
    return n_dot * M / (rho * A(d) * 1000)

def M_mix(x_arr):
    x_WA = x_arr[0]
    x_EG = x_arr[1]
    x_EO = x_arr[2]
    return x_WA * M_WA + x_EG * M_EG + x_EO * M_EO

# Locations of the injections based on the length of the system and the distance between injections, initial injection not included
def injection_locations(L, N):
    delta_L = L / (N + 1)
    loc = []
    for i in range(1, N + 1):
        loc.append(i*delta_L)
    return loc

            



    

    

    
def initial_injection(n_dot_in_EO , N):
    return n_dot_in_EO / (N + 1) # Initial flow rate of EO at the first injection point


T_wall = 423 # Wall temperature in K
c_10 = 2270 # Initial concentration of EO in mol/m^3
c_20 = 0 # Initial concentration of EG in mol/m^3
T_0 = 423 # Initial temperature in K

u_ref = 1 # Reference velocity in m/s

A_ref = A(d_R) # Cross-sectional area of the reactor in m^2


def search_WA_inlet(n_dot_in_WA, n_dot_in_EO, T_0):
    x_WA = n_dot_in_WA / (n_dot_in_WA + n_dot_in_EO)
    x_EO = n_dot_in_EO / (n_dot_in_WA + n_dot_in_EO)

    x_arr = np.array([x_WA[0], 0, x_EO[0]])

    u_ = u(n_dot_in_WA + n_dot_in_EO, rho_mix(T_0, x_arr), d_R, M_mix(x_arr))

    return u_ - u_ref

delta_x = 0.01

#%%

def PFTR_thermodynamics(L, f):
    """
    Spatial derivatives for the PFTR material and energy balances.

    f[0] = c_1  (EO concentration, mol/m^3)
    f[1] = c_2  (EG concentration, mol/m^3)
    f[2] = T    (temperature, K)

    Space-time transformation:  d/dL = (1/u) * d/dt
    Each right-hand side is the bSTR time-derivative divided by u.
    """
    c_1 = f[0]
    c_2 = f[1]
    c_3 = f[2]
    T   = f[3]

    c_arr = np.array([c_1, c_2, c_3])

    x_1 = x_i(c_arr, c_1)
    x_2 = x_i(c_arr, c_2)
    x_3 = x_i(c_arr, c_3)

    x_arr = np.array([x_1, x_2, x_3])


    r = kinetics(T, c_1)


    # Component balances (material balance per unit reactor length)
    dc_1dL = -r / u_ref
    dc_2dL =  r / u_ref
    dc_3dL = -r / u_ref

    C = c_p_mix(T, x_arr) * rho_mix(T, x_arr)

    # Energy balance per unit reactor length
    # Reaction term:    -H_R * r / C              [K/s] / [m/s] = K/m
    # Heat-transfer:    h_wall * a_wall * (T-T_wall) / C  [K/m]
    dTdL = (-H_R * r / C - h_wall * a_wall * (T - T_wall) / C) / u_ref

    dfdL = np.empty_like(f)
    dfdL[0] = dc_1dL
    dfdL[1] = dc_2dL
    dfdL[2] = dc_3dL
    dfdL[3] = dTdL
    return dfdL

#%%

x_prev = 0

Lspan = np.array([0, L_R])
leval = np.linspace(0, Lspan[1], 2001)

def sol_thermodynamics(c_arr, T_0):
    c_EO = c_arr[0]
    c_EG = c_arr[1]


    n_dot_in_EO_tot = c_EO * u_ref * A_ref # Total molar flow rate of EO in mol/s
    n_dot_in_EG_tot = c_EG * u_ref * A_ref # Total molar flow rate of EO in mol/s

    n_dot_in_WA_initial_guess = n_dot_in_EO_tot # Initial guess for the molar flow rate of water in mol/s    

    n_dot_in_WA = root(search_WA_inlet, n_dot_in_WA_initial_guess, args=(n_dot_in_EO_tot, T_0)).x[0] # Solve for the molar flow rate of water in mol/s

    c_WA = n_dot_in_WA / (u_ref * A_ref)

    initial = np.array([c_EO, c_EG, c_WA, T_0])

    sol_thermodynamic = solve_ivp(PFTR_thermodynamics, Lspan, initial, method='BDF', t_eval=leval)

    return sol_thermodynamic.t, sol_thermodynamic.y[0], sol_thermodynamic.y[1], sol_thermodynamic.y[2], sol_thermodynamic.y[3] 

sol_th = sol_thermodynamics(np.array([c_10, c_20]), T_0)

fig_th, ax_th = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

ax_th[0].plot(sol_th[0], sol_th[1])
ax_th[0].plot(sol_th[0], sol_th[2])
ax_th[1].plot(sol_th[0], sol_th[4])

def PFTR_inj(x, f):
    n_dot_WA = f[0]
    n_dot_EG = f[1]
    n_dot_EO = f[2] 
    T = f[3]

    x_WA = n_dot_WA / (n_dot_WA + n_dot_EG + n_dot_EO)
    x_EG = n_dot_EG / (n_dot_WA + n_dot_EG + n_dot_EO)
    x_EO = n_dot_EO / (n_dot_WA + n_dot_EG + n_dot_EO)

    rho_ = rho_mix(T, np.array([x_WA, x_EG, x_EO]))
    c_p = c_p_mix(T, np.array([x_WA, x_EG, x_EO])) # (M_mix(np.array([x_WA, x_EG, x_EO])) * 10e-3)


    u_ = u(n_dot_WA + n_dot_EG + n_dot_EO, rho_, d_R, M_mix(np.array([x_WA, x_EG, x_EO])))

    r = kinetics(T, n_dot_EO / (A(d_R)*u_))

    dn_EO_dx = -r * A_ref
    dn_EG_dx = r * A_ref
    dn_WA_dx = -r * A_ref

    dT_dx = (-H_R * r) / (rho_ * c_p * u_) - 0.25 * h_wall * a_wall * (T - T_wall) / (rho_ * c_p * u_)

    dfdx = np.empty_like(f)
    dfdx[0] = dn_WA_dx
    dfdx[1] = dn_EG_dx
    dfdx[2] = dn_EO_dx
    dfdx[3] = dT_dx
    x_prev = x
    
    return dfdx


x_prev = 0
L_arr = np.arange(0, L_R, delta_x)



def sol_inject(c_arr, T_0, N):
    c_EO = c_arr[0]
    c_EG = c_arr[1]


    n_dot_in_EO_tot = c_EO * u_ref * A_ref # Total molar flow rate of EO in mol/s
    n_dot_in_EG_tot = c_EG * u_ref * A_ref # Total molar flow rate of EO in mol/s

    x_arr_inj = injection_locations(L_R, N)
    n_dot_in_EO_injection = n_dot_in_EO_tot / (N + 1) # Flow rate of EO at each injection point

    arr = {}

    delta_L = L_R / (N + 1)

    for i, x_i in enumerate(x_arr_inj):
        t_arr = np.arange(i*delta_L, (i+1)*delta_L, 0.1)
        arr.update({f"{i}": t_arr})
    numb = N 
    arr.update({f"{numb}": np.arange(numb*delta_L, (numb+1)*delta_L, 0.1)})
    n_dot_in_EO = initial_injection(n_dot_in_EO_tot, N) # Molar flow rate of EO at each injection point in mol/s

    n_dot_in_WA_initial_guess = n_dot_in_EO # Initial guess for the molar flow rate of water in mol/s    

    n_dot_in_WA = root(search_WA_inlet, n_dot_in_WA_initial_guess, args=(n_dot_in_EO, T_0)).x[0] # Solve for the molar flow rate of water in mol/s

    n_dot_WA_out = np.array([n_dot_in_WA])
    n_dot_EG_out = np.array([0])
    n_dot_EO_out = np.array([0])
    T_inj = np.array([T_0])
    t = np.array([0])

    for i in arr.keys():
        print(i)
        L_arr = arr[f"{i}"]

        n_tot = n_dot_WA_out[-1] + n_dot_EG_out[-1] + n_dot_EO_out[-1]

        x_in =  n_dot_in_EO / n_tot

        initial = np.array([n_dot_WA_out[-1], n_dot_EG_out[-1], n_dot_EO_out[-1] + n_dot_in_EO , T_inj[-1] * (1 - x_in) + T_0 * x_in])
        sol_inj = solve_ivp(PFTR_inj, np.array([L_arr[0], L_arr[-1]]), initial, t_eval=L_arr, method="BDF", max_step=0.1)

        t = np.append(t , sol_inj.t)
        n_dot_WA_out = np.append(n_dot_WA_out, sol_inj.y[0])
        n_dot_EG_out = np.append(n_dot_EG_out, sol_inj.y[1])
        n_dot_EO_out = np.append(n_dot_EO_out, sol_inj.y[2])
        T_inj = np.append(T_inj, sol_inj.y[3])

    x_WA_inj_out = n_dot_WA_out / (n_dot_WA_out + n_dot_EO_out + n_dot_EG_out)
    x_EO_inj_out = n_dot_EO_out / (n_dot_WA_out + n_dot_EO_out + n_dot_EG_out)
    x_EG_inj_out = n_dot_EG_out / (n_dot_WA_out + n_dot_EO_out + n_dot_EG_out)

    u_ = u(n_dot_WA_out+n_dot_EG_out+n_dot_EO_out, rho_mix(T_inj, np.array([x_WA_inj_out, x_EG_inj_out, x_EO_inj_out])), d_R, M_mix(np.array([x_WA_inj_out, x_EG_inj_out, x_EO_inj_out])))

    print(x_arr_inj)
    return t, n_dot_WA_out / (u_ * A_ref), n_dot_EG_out / (u_ * A_ref), n_dot_EO_out / (u_ * A_ref), T_inj, u_

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

sol = sol_inject(np.array([c_10, c_20]), 450, 3)

ax[0].plot(sol[0], sol[2])
ax[0].plot(sol[0], sol[3], color="blue")
ax[1].plot(sol[0], sol[4])



plt.show()



# %%
