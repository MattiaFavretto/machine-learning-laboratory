from jax import vmap
from jax import numpy as np

# FUNZIONA

a = np.array([1,2])

def compute_arctan(x):
    return np.arctan(x)

compute_arctan_v = vmap(compute_arctan, in_axes=(0)) # il numero qui indica l'asse sul quale si trova il nostro vettore di input

result = compute_arctan_v(a)
#result = compute_arctan(a)
print(result)

# prova a farlo con una sola dimensione
