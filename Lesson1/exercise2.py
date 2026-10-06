from jax import grad
from jax import numpy as np


# Come mai non funziona con numeri interi????

a = 0.5

def compute_arctan(x):
    return np.arctan(x)

compute_actan_grad = grad(compute_arctan, argnums=(0), allow_int=True)
result = compute_actan_grad(a)
print(result)
print((1/(1+a*a)))
