from jax import jit, rnd, numpy as jnp

seed = 1701
k = rnd.key(seed)
input = rnd.normal(k, (4, 2))

input = jnp.array([[0.0,1.0],[2.0,3.0],[3.0,4.0],[5.0,-1.0]]) 

def dense_layer(x):
    # input: array containing both inputs and weights
    sum = 0
    for i in range(len(x)):
        sum = sum + x[i][0]*x[i][1]
    # Applying the ReLU activation function with the max
    return jnp.maximum(0,sum)

dense_layer_jit = jit(dense_layer)
result = dense_layer_jit(input)
print(result)
