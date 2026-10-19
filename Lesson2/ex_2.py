from jax import numpy as jnp, random as rnd, vmap,jit,lax
from jax.scipy.stats import multivariate_normal # it is used to calculate automatically the pdf of the gaussian
import pandas as pd
import timeit
import matplotlib.pyplot as plt

# importing datas
df = pd.read_csv('old_faithful.tsv', sep='\t')
x = jnp.array(df.values)

# create random values for the two gaussian
seed = 1701
k1,k2,key = rnd.split(rnd.key(seed),3) # unless they will have the same value

num_features = x.shape[1]
num_gauss = 2
num_iterations = 100
responsability = jnp.zeros((len(x),num_gauss))

# I create a dictionary of arrays rappresenting the parameters of the two gaussians
# I preferred it instead of a list of dictionaries because the first one is more efficient
# It is also simplier to add a new gaussian, it is just needed to add a value in every array

# selected two vector among the input to have a better starting point for the gaussians
idx = rnd.choice(k1, len(x), shape=(num_gauss,), replace=False)

gauss = {
    "mean" : x[idx],
    "covariance" : jnp.array([jnp.cov(x, rowvar=False),jnp.cov(x, rowvar=False)]), # used the covariance of the whole dataset to have a better starting point and avoid NaN values
    "weigth" : jnp.ones(num_gauss) / num_gauss, # in order to make the code reusable for different inputs
}

def responsability_n(x, gauss):
    tmp = []
    for j in range(num_gauss):
        # with 2 gaussians can also calculate resp_1 and then use 1-resp_1 but I preferred to calculate it every time to generalize the function
        tmp.append(gauss["weigth"][j] * multivariate_normal.pdf(x, mean=gauss["mean"][j], cov=gauss["covariance"][j]))

    result = jnp.array(tmp)
    result = result/jnp.sum(result)
    return result

responsability_vmap = vmap(responsability_n,in_axes=(0,None))
responsability_jit = jit(responsability_vmap)

def compute_covariance(responsability, x, gauss):
    tmp = []
    for j in range(num_gauss):
        tmp.append(responsability[j] * (jnp.outer(jnp.transpose(x-gauss["mean"][j]),x-gauss["mean"][j])))

    return jnp.array(tmp)

compute_covariance_vmap = vmap(compute_covariance,in_axes=(0,0,None))
compute_covariance_jit = jit(compute_covariance_vmap)

def one_step_em(x,responsability,gauss,N):
    # calculate the likelihood of a point
    responsability = responsability_jit(x,gauss)

    # recalculate the two gaussian distributions based on the new calculated likelihood
    # 1. Recalculate the weigth
    gauss["weigth"] = jnp.sum(responsability,0)/N # jnp.sum(,0) calculates the sum considering the first dimension, so for every column calculates the total sum

    # 2. Recalculate the mean
    gauss["mean"]= jnp.transpose(responsability) @ x/jnp.sum(responsability,0)[:,None] # in order to have one column to divide the matrix column by column

    # 3. Recalculate the covariance
    gauss["covariance"] = jnp.sum(compute_covariance_jit(responsability,x,gauss),0)/jnp.sum(responsability,0)[:,None,None]

    return gauss, responsability

one_step_em_jit = jit(one_step_em)

def compute_q(x,responsability,gauss):
    tmp = []
    for j in range(num_gauss):
        tmp.append(responsability[j] * (jnp.log(gauss["weigth"][j]) + multivariate_normal.logpdf(x, mean=gauss["mean"][j], cov=gauss["covariance"][j])))

    return jnp.sum(jnp.array(tmp))

compute_q_vmap = vmap(compute_q, in_axes=(0,0,None))
compute_q_jit = jit(compute_q_vmap)

def expectation_maximization(x,gauss,responsability):

    N = len(x)
    # Q_old = -jnp.inf
    # tollerance = 1e-4

    # for loop for the em
    for y in range(num_iterations):
        gauss, responsability = one_step_em_jit(x,responsability,gauss,N)

        # COMMENTED BECAUSE IN POINT C) TO COMPILE THE FUNCTION WITH JIT WE CANNOT HAVE IF INSIDE
        # stop the cycle when the difference is too small
        # Q_new = jnp.sum(compute_q_jit(x,responsability,gauss))
        # if jnp.abs(Q_new-Q_old) < tollerance:
        #     break

        # Q_old = Q_new
    return gauss

# SOLUTION 1 - use of for cicle and then jit
expectation_maximization_jit = jit(expectation_maximization)

# Call the function the first time to compile everything
# starting_prep_time = timeit.default_timer()
# _ = expectation_maximization_jit(x,gauss,responsability)["mean"].block_until_ready() # it is used to return the function when every computation is ended
# final_prep_time = timeit.default_timer()
# first_execution_time = final_prep_time - starting_prep_time
# print("Time of the first execution", first_execution_time)

# Call the function again
# starting_time = timeit.default_timer()
# gauss = expectation_maximization_jit(x,gauss,responsability)
# gauss["mean"].block_until_ready()
# final_time = timeit.default_timer()
# execution_time = final_time - starting_time

# print("Time for the execution", execution_time)

# SOLUTION 2 - use of jax.lax.fori_loop

def expectation_maximization_fori(x, gauss, responsability):
  N = len(x)
  num_iterations = 100

  # Define a function that receive only two arguments and one output, as fori_loop requires
  def body_fun(i, input):
    gauss, responsability = input
    gauss, responsability = one_step_em_jit(x, responsability, gauss, N)
    return (gauss, responsability)

  gauss, responsability = lax.fori_loop(0, num_iterations, body_fun, (gauss, responsability))

  return gauss

expectation_maximization_fori_jit = jit(expectation_maximization_fori)

# Call the function the first time to compile everything
starting_prep_time = timeit.default_timer()
_ = expectation_maximization_fori_jit(x,gauss,responsability)["mean"].block_until_ready() # it is used to return the function when every computation is ended
final_prep_time = timeit.default_timer()
first_execution_time = final_prep_time - starting_prep_time
print("Time of the first execution", first_execution_time)


# Call the function again
starting_time = timeit.default_timer()
gauss = expectation_maximization_fori_jit(x,gauss,responsability)
gauss["mean"].block_until_ready()
final_time = timeit.default_timer()
execution_time = final_time - starting_time

print("Time for the execution", execution_time)
# Note: the execution time is way faster, on the first execution and also on the others

# PRINT THE RESULTS
colors = ['red','green']
plt.figure(figsize=(10, 10))
# Plot the points
plt.scatter(x[:, 0], x[:, 1], s=15)

# Create a circle
theta = jnp.linspace(0,2*jnp.pi,100)
unit_circle = jnp.array([jnp.cos(theta),jnp.sin(theta)])

for j in range(num_gauss):

    # Plot the mean of the gaussian
    plt.scatter(gauss["mean"][j][0], gauss["mean"][j][1], color=colors[j], marker='X', s=100, label=f'Mean Gaussian {j+1}')


    L = jnp.linalg.cholesky(gauss["covariance"][j])
    for k in [1,2,3]:
        ellipse = gauss["mean"][j][:,None] + k * (L @ unit_circle)
        # Plot the ellipse formed by 100 point joint with a straight line
        plt.plot(ellipse[0, :], ellipse[1, :], color=colors[j], linewidth=1.5)

plt.xlabel("Eruption duration")
plt.ylabel("Waiting time")
plt.legend()
plt.show()
