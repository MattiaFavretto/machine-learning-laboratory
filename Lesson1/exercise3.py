from jax import jit
from jax import numpy as jnp

# attenzione che jnp.max cerca il massimo in un array, non fra due elementi

input = jnp.array([[0,1],[2,3],[3,4],[5,-1]])

def dense_layer(x):
    # input: x array che contiene sia inout che pesi
    sum = 0
    # print(len(x))
    for i in range(len(x)):
        # print(x[i][0])
        # print(x[i][1])
        sum = sum + x[i][0]*x[i][1]
    # print(sum)
    return max(0,sum)

# TODO - FINIRE
# result = dense_layer(input)
dense_layer_jit = jit(dense_layer, )
result = dense_layer_jit(input)
print(result)
