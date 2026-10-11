from jax import vmap
from jax import numpy as np

inp = np.array([1,2])

def compute_arctan(x):
    return np.arctan(x)

compute_arctan_v = vmap(compute_arctan, in_axes=(0))
result = compute_arctan_v(inp)

print(result)

