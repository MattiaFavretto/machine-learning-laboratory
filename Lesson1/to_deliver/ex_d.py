from jax import jit, random as rnd, grad, numpy as jnp, vmap
import numpy as np
import matplotlib.pyplot as plt

# DECLARING VARIABLES
seed = 1701
learning_rate = 0.001
tolerance = 1e-10
epochs = 200
prev_loss = float("inf")
error_list = [] # list of the errors to see the progression

# LOADING THE I/O FILES
x_raw = np.load("data_x.npy")
y_raw = np.load("data_y.npy")

# convert the input into an array jax numpy
x = jnp.array(x_raw)
y_true = jnp.array(y_raw).flatten() # used to transform the matrix into a 1-dimensional array

# creation of the random weights

k1 = rnd.key(seed)
k2 = rnd.key(seed)
k3 = rnd.key(seed)

w1 = rnd.normal(k1,(2,4))
# output of first layer is a (1,4) matrix
w2 = rnd.normal(k2,(4,4))
# output of first layer is a (4,1) matrix
w3 = rnd.normal(k3,(4,1))
# output of first layer is a number

# NEURAL NETWORK CREATION
# I created a 3 layer neural network with 4 neurons in the hidden layers
def neural_network(x,w1,w2,w3):
    # calculate the output of the first layer
    out1 = []
    for y in range(4):
        sum = 0
        for z in range(2):
            sum = sum + x[z]*w1[z][y]
        out1.append(jnp.maximum(0,sum))
    # calculate the output of the second layer
    out2 = []
    for y in range(4):
        sum = 0
        for z in range(4):
            sum = sum + out1[z]*w2[z][y]
        out2.append(jnp.maximum(0,sum))

    # calculate the output of the third layer, it does not have the activation function, it does have only one node
    sum = 0
    for y in range(4):
        sum = sum + out2[y]*w3[y][0]

    return sum

network_vmap = vmap(neural_network,in_axes=(0,None,None,None))
# create the least squared loss function over the hole input
def loss_function(input,y_true,w1,w2,w3):
    y_pred = network_vmap(input,w1,w2,w3)
    return jnp.mean((y_pred-y_true)**2) # I calculate the mean because I want only one number in output, not an array

loss_function_grad = grad(loss_function,argnums=(2,3,4))

def update_weights(x,y_true,w1,w2,w3):
    error = loss_function(x,y_true,w1,w2,w3)
    # calculate the gradient of the error function based on the weights
    gradient_w1, gradient_w2, gradient_w3 = loss_function_grad(x,y_true,w1,w2,w3)
    # update the weights using the gradient and the learning rate
    w1 = w1 - learning_rate*gradient_w1
    w2 = w2 - learning_rate*gradient_w2
    w3 = w3 - learning_rate*gradient_w3
    return error,w1,w2,w3

update_weights_jit = jit(update_weights)

for i in range(epochs):
    error,w1,w2,w3 = update_weights_jit(x,y_true,w1,w2,w3)
    current_loss = error
    error_list.append(current_loss)

    if abs(prev_loss - current_loss) < tolerance:
        break
    prev_loss = current_loss


# PLOTTING RESULTS
plt.figure(figsize=(10,10))
plt.plot(error_list)
plt.grid(True)

plt.show()
