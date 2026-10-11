from jax import jit, random as rnd, numpy as jnp

seed = 1701
k1 = rnd.key(seed)
k2 = rnd.key(seed)

inp = rnd.normal(k1, (10,))
weight_bias = rnd.normal(k2, (10, 2))

def dense_layer(x,w):
    # input: array containing input and weight/bias
    sum = 0
    for i in range(len(x)):
        sum = sum + x[i]*w[i][1]
    sum = sum + w[0][0] # adding the bias
    # Applying the ReLU activation function with the max
    return jnp.maximum(0,sum)

dense_layer_jit = jit(dense_layer)
result = dense_layer_jit(inp, weight_bias)
print(result)
