from jax import jit
from jax import numpy as jnp

# attenzione che jnp.max cerca il massimo in un array, non fra due elementi
# ATTENZIONE, NON POSSO USARE FUNZIONI CLASSICHE DI PYTHON, NEMMENO COME MAX

input = jnp.array([[0,1],[2,3],[3,4],[5,-1]])

def dense_layer(x):
    # input: array containing both inputs and weights
    sum = 0
    for i in range(len(x)):
        sum = sum + x[i][0]*x[i][1]
    # Applying the ReLU activation function with the max
    return jnp.maximum(0,sum)

# result = dense_layer(input)
dense_layer_jit = jit(dense_layer)
result = dense_layer_jit(input)
print(result)
