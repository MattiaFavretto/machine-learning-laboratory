from jax import numpy as jnp, random as rnd, vmap, jit, lax
from jax.scipy.stats import multivariate_normal
import pandas as pd
import timeit
import matplotlib.pyplot as plt
import data_loader_speech as dl


# importing datas
sentence, transcriptions = next(dl.data_iter())

# create random values for the two gaussian
seed = 1701
k1,k2,key = rnd.split(rnd.key(seed),3) # unless they will have the same value

_, phonemes_list = dl.get_phonemes()
num_features = sentence.shape[1] # the number of features is the dimension of the input
num_gauss = len(phonemes_list) # the is a gaussian for every phoneme
num_iterations = 100
num_frame = sentence.shape[0] # the number of frames, so the hidden states of our HMM
responsability = jnp.zeros((num_frame,num_gauss))

idx = rnd.choice(k1, num_frame, shape=(num_gauss,), replace=True)

# parameters of the gaussians
mean = sentence[idx] # I take the value of the component of the first sentence as initial value
covariance = jnp.tile(jnp.cov(sentence, rowvar=False)[None,:,:],(num_gauss,1,1)) # used to repeat the covariance for every gaussian, for the base version it is initialized using only the first sentence. The book suggest to initialize it in another way (pag. 623)

# as asked I initialized the transition matrix and the initial parameters to be equally likely
pi = jnp.ones(num_gauss)/num_gauss
A = jnp.ones((num_gauss,num_gauss))/num_gauss

# Calculate the initial alpha and final beta
initial_alpha = pi * multivariate_normal.pdf(sentence[0],mean=mean,cov=covariance)
final_beta = jnp.ones((num_gauss,))
c_0 = jnp.sum(initial_alpha)
initial_alpha = initial_alpha/c_0

def compute_covariance(responsability, x, mean):
    tmp = []
    for j in range(num_gauss):
        tmp.append(responsability[j] * (jnp.outer(jnp.transpose(x-mean[j]),x-mean[j])))

    return jnp.array(tmp)

compute_covariance_vmap = vmap(compute_covariance,in_axes=(0,0,None))
compute_covariance_jit = jit(compute_covariance_vmap)

# input: x are the evidence
def one_step_em(x,mean,covariance,initial_alpha,c_0,final_beta,A):
    # defined inside of the one_step_em function to access gauss and A
    def forward_step(carry,x):
        # the forward_step function need to respect the scan function requirement
        old_alpha = carry

        # I need to repeat this for every k
        new_alpha = multivariate_normal.pdf(x,mean=mean,cov=covariance) * (old_alpha@A)
        new_c = jnp.sum(new_alpha)
        new_alpha = new_alpha / new_c

        return new_alpha, (new_alpha, new_c)

    def backaward_step(carry,input):
        # the backward_step function need to respect the scan function requirement
        old_beta = carry
        x, c = input
        # I need to repeat this for every k
        new_beta = A @ (multivariate_normal.pdf(x,mean=mean,cov=covariance) * old_beta)
        new_beta = new_beta / c

        return new_beta, new_beta

    def compute_step_csi(c,x,alpha,beta):
        csi_n = (jnp.outer(alpha,multivariate_normal.pdf(x,mean=mean,cov=covariance) * beta) * A )/c
        return csi_n

    compute_csi = vmap(compute_step_csi,in_axes=(0,0,0,0))


    # Calculate the forward step
    _,output_forward = lax.scan(forward_step,initial_alpha,x[1:],reverse=False)
    alpha, c = output_forward
    alpha = jnp.vstack([initial_alpha[None, :], alpha])
    c = jnp.concatenate(jnp.array([c]),c_0)

    # Calculate the backward step
    _,beta = lax.scan(backaward_step,final_beta,(x[1:],c[1:]),reverse=True) # it is [1:] because the reverse starts form the last element
    beta = jnp.vstack([beta, final_beta[None, :]])


    # Recalculate every parameter given the new alpha and beta

    # responsability
    responsability = alpha*beta
    # transition probability
    csi = compute_csi(c[1:],x[1:],alpha[:-1],beta[1:])
    # pi
    pi = responsability[0]
    csi_sum = jnp.sum(csi,axis=(0))
    A = csi_sum/jnp.sum(csi_sum,axis=(1))[:,None]
    # recalculate the gaussian distributions
    # 1. Recalculate the mean
    mean = jnp.transpose(responsability) @ x/jnp.sum(responsability,0)[:,None] # in order to have one column to divide the matrix column by column
    # 2. Recalculate the covariance
    covariance = jnp.sum(compute_covariance_jit(responsability,x,mean),0)/jnp.sum(responsability,0)[:,None,None]

    # Prepare for the next iteration
    initial_alpha = pi * multivariate_normal.pdf(x[0],mean=mean,cov=covariance)
    final_beta = jnp.ones((num_gauss,))
    c = jnp.sum(initial_alpha)
    initial_alpha = initial_alpha/c


    return mean, covariance, pi, initial_alpha, final_beta, A

one_step_em_jit = jit(one_step_em)


# def compute_q(x,responsability,gauss):
#     tmp = []
#     for j in range(num_gauss):
#         tmp.append(responsability[j] * (jnp.log(gauss["weigth"][j]) + multivariate_normal.logpdf(x, mean=mean[j], cov=covariance[j])))

#     return jnp.sum(jnp.array(tmp))

# compute_q_vmap = vmap(compute_q, in_axes=(0,0,None))
# compute_q_jit = jit(compute_q_vmap)

# Usage of the function
def expectation_maximization_fori(x, mean, covariance, initial_alpha, final_beta, A):

    num_iterations = 100

        # Define a function that receive only two arguments and one output, as fori_loop requires
        def body_fun(i, input):
            x, mean, covariance, initial_alpha, final_beta, A = input
            mean, covariance, initial_alpha, final_beta, A = one_step_em_jit(x, mean, covariance, initial_alpha, c_0, final_beta, A)
            return (mean, covariance, initial_alpha, final_beta, A)

    mean, covariance, pi_0, _, _, A = lax.fori_loop(0, num_iterations, body_fun, (x, mean, covariance, initial_alpha, c_0, final_beta, A))

    return mean, covariance, pi_0, A

# # Call the function the first time to compile everything
# starting_prep_time = timeit.default_timer()
# _ = expectation_maximization_fori(num_frame,gauss,responsability)["mean"].block_until_ready() # it is used to return the function when every computation is ended
# final_prep_time = timeit.default_timer()
# first_execution_time = final_prep_time - starting_prep_time
# print("Time of the first execution", first_execution_time)


# Call the function again
starting_time = timeit.default_timer()
mean, covariance, pi_0, A = expectation_maximization_fori(sentence, mean, covariance, initial_alpha, final_beta, A)
mean.block_until_ready()
final_time = timeit.default_timer()
execution_time = final_time - starting_time

print("Time for the execution", execution_time)
# Note: the execution time is way faster, on the first execution and also on the others

# PRINT THE RESULTS
