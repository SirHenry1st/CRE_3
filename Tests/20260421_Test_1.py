#%%
import numpy as np
import matplotlib.pyplot as plt


#%%

a = 1
b = 2

def parabel (x):
    y = a * np.power(x, 2) + b
    return y


#%%

x_vec = np.linspace(-5, 5, 150)

fig, ax = plt.subplots()
ax.plot(x_vec, parabel(x_vec))
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show
