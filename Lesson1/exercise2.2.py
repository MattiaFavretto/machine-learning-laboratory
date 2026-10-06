from jax import jacobian
from jax import numpy as np

# funzione grad - calcola la derivata di una funzione (gradiente) rispetto ai suoi input
# argnums indica solamente rispetto a quale input andare a calcolare il gradiente, ma se l'input di una funzione e' un vettore lo fa rispetto a tutti i valori


# GLI INPUT DEVONO ESSERE DEI NUMERI REALI altrimenti uso ALLOW_INT
a = np.array([1.0,2.0,3.0,4.0])

def compute_arctan(x):
    return np.arctan(x)


compute_actan_jacobian = jacobian(compute_arctan, argnums=(0), allow_int=True)
result = compute_actan_jacobian(a)
print(result)


#return 1/(1+t*t)
