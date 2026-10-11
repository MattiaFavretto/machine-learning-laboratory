import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import grad, jit, vmap, numpy as jnp, random as rnd

# declaring variables 

learning_rate = 0.001
seed = 1701
tolerance = 1e-10
epochs = 1000
prev_loss = float("inf")
error_list = []

# importing datas
x_raw = np.load("data_x.npy")
y_raw = np.load("data_y.npy")

x = jnp.array(x_raw)
y_true = jnp.array(y_raw).flatten()


k1 = rnd.key(seed)
k2 = rnd.key(seed)
k3 = rnd.key(seed)

w1 = rnd.normal(k1, (2, 4))
w2 = rnd.normal(k2, (4, 4))
w3 = rnd.normal(k3, (4, 1))
b1 = jnp.zeros((4,))
b2 = jnp.zeros((4,))
b3 = jnp.zeros((1,))



def neural_network(x, w1, w2, w3, b1, b2, b3):

    out1 = []
    for y in range(4):
        sum = 0
        for z in range(2):
            sum = sum + x[z] * w1[z][y]
        sum = sum + b1[y]
        out1.append(jnp.maximum(0, sum))

    out2 = []
    for y in range(4):
        sum = 0
        for z in range(4):
            sum = sum + out1[z] * w2[z][y]
        sum = sum + b2[y]
        out2.append(jnp.maximum(0, sum))
    sum = 0
    for y in range(4):
        sum = sum + out2[y] * w3[y][0]
    sum = sum + b3[0]

    return sum

network_vmap = vmap(neural_network, in_axes=(0, None, None, None, None, None, None))


def loss_function(input, y_true, w1, w2, w3, b1, b2, b3):
    y_pred = network_vmap(input, w1, w2, w3, b1, b2, b3)
    return jnp.mean((y_pred - y_true) ** 2)

loss_function_grad = grad(loss_function, argnums=(2, 3, 4, 5, 6, 7))

# using the adam optimizer
optimizer = optax.adam(learning_rate=learning_rate) # create the optimazer for this learning rate
opt_state = optimizer.init((w1, w2, w3, b1, b2, b3)) # initialize the optimazer



def update_weights(x, y_true, w1, w2, w3, b1, b2, b3, opt_state):
    error = loss_function(x, y_true, w1, w2, w3, b1, b2, b3)
    gradient_w1, gradient_w2, gradient_w3, gradient_b1, gradient_b2, gradient_b3 = loss_function_grad(x, y_true, w1, w2, w3, b1, b2, b3)
    gradients = (gradient_w1, gradient_w2, gradient_w3, gradient_b1, gradient_b2, gradient_b3)
    # Calcola gli update e aggiorna opt_state (non scartarlo con _)
    updates, opt_state = optimizer.update(gradients, opt_state, (w1, w2, w3, b1, b2, b3))

    # Applica gli aggiornamenti
    w1, w2, w3, b1, b2, b3 = optax.apply_updates((w1, w2, w3, b1, b2, b3), updates)

    return error, w1, w2, w3, b1, b2, b3, opt_state

update_weights_jit = jit(update_weights)

for i in range(epochs):
    error, w1, w2, w3, b1, b2, b3, opt_state = update_weights_jit(x, y_true, w1, w2, w3, b1, b2, b3, opt_state)
    current_loss = error
    error_list.append(current_loss)

    if abs(prev_loss - current_loss) < tolerance:
        break

    prev_loss = current_loss

# plotting the result
plt.figure(figsize=(10, 10))
plt.plot(error_list)
plt.grid(True)

plt.show()
