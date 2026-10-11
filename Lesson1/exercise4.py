from jax import jit, random as rnd, grad, numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# attenzione che jnp.max cerca il massimo in un array, non fra due elementi
# ATTENZIONE, NON POSSO USARE FUNZIONI CLASSICHE DI PYTHON, NEMMENO COME MAX
# ATTENZIONE, IN JAX GLI ARRAY SONO IMMUTABILI, devo usare delle liste

# Specification: I used all the data as training set
# list of the errors to see the progression
error_list = []

# loading the input file and the output file
x_raw = np.load("data_x.npy")
y_raw = np.load("data_y.npy")

# convert the input into an array jax numpy
x = jnp.array(x_raw)
y_true = jnp.array(y_raw).flatten() # used to transform the matrix into a 1-dimensional array

# creation of the random weights
seed = 1701
k = rnd.key(seed)

w1 = rnd.normal(k,(2,4))
# output of first layer is a (1,4) matrix
w2 = rnd.normal(k,(4,4))
# output of first layer is a (4,1) matrix
w3 = rnd.normal(k,(4,1))
# output of first layer is a number

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

    # calculate the output of the third layer, it does not have the activation function
    # it does have only one node
    sum = 0
    for y in range(4):
        sum = sum + out2[y]*w3[y][0]

    return sum


# create the least squared loss function
def least_squared_loss_function(input,y_true,w1,w2,w3):
    y_pred = neural_network(input,w1,w2,w3)
    return (y_pred-y_true)**2



least_squared_loss_function_grad = grad(least_squared_loss_function,argnums=(2,3,4))


learning_rate = 0.001
tolerance = 1e-10
epochs = 200
prev_loss = float("inf")
for i in range(epochs):
    epoch_loss = 0.0
    for y in range(100):
        error = least_squared_loss_function(x[y],y_true[y],w1,w2,w3)
        epoch_loss += error
        # calculate the gradient of the error function based on the weights
        gradient_w1, gradient_w2, gradient_w3 = least_squared_loss_function_grad(x[y],y_true[y],w1,w2,w3)
        # update the weights using the gradient and the learning rate
        w1 = w1 - learning_rate*gradient_w1
        w2 = w2 - learning_rate*gradient_w2
        w3 = w3 - learning_rate*gradient_w3
    current_loss = epoch_loss / 100
    error_list.append(current_loss)

    if(prev_loss - current_loss)<tolerance:
        break

    prev_loss = current_loss
print(error_list)


# Plotting the results
plt.figure(figsize=(10,10))
plt.plot(error_list)
plt.grid(True)

plt.show()
