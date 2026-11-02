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
phoneme2idx = {p: i for i, p in enumerate(phonemes_list)} # I convert phonems to index because I need that to use the evidence in the e_step
num_gauss = len(phonemes_list) # the is a gaussian for every phoneme

num_features = sentence.shape[1] # the number of features is the dimension of the input
num_frame = sentence.shape[0] # the number of frames, so the hidden states of our HMM


idx = rnd.choice(k1, num_frame, shape=(num_gauss,), replace=True)

# parameters of the gaussians
mean = sentence[idx] # I take the value of the component of the first sentence as initial value
covariance = jnp.tile(jnp.cov(sentence, rowvar=False)[None,:,:],(num_gauss,1,1)) # used to repeat the covariance for every gaussian, for the base version it is initialized using only the first sentence. The book suggest to initialize it in another way (pag. 623)


# as asked I initialized the transition matrix and the initial parameters to be equally likely
A = jnp.ones((num_gauss,num_gauss))/num_gauss


def compute_covariance(responsability, x, mean):
    tmp = []
    for j in range(num_gauss):
        tmp.append(responsability[j] * (jnp.outer(jnp.transpose(x-mean[j]),x-mean[j])))

    return jnp.array(tmp)

compute_covariance_vmap = vmap(compute_covariance,in_axes=(0,0,None))
compute_covariance_jit = jit(compute_covariance_vmap)

# input: x are the evidence
def step_e(x,y,mean,covariance):
    # Get di length (n) of the sentence and map the indeces of sentence phonemes considering the global list of phonems
    n = len(y)
    y_index = jnp.array([phoneme2idx[p] for p in y])

    # Initialize alfa and beta inside the function to deal also with sentences with different length
    pi = jnp.zeros(n).at[0].set(1.0)
    initial_alpha = pi * multivariate_normal.pdf(x[0],mean=mean[y_index[0]],cov=covariance[y_index[0]])
    c_0 = jnp.sum(initial_alpha)
    initial_alpha = initial_alpha/c_0

    final_beta = jnp.zeros(n).at[-1].set(1.0)

    # Create the A matrix for this situation A_sub[i, j] = A[p(i), p(j)]
    A_sub = A[y_index[:, None], y_index[None, :]]

    # maske to only have left-to-right
    mask = jnp.eye(n) + jnp.eye(n, k=1)
    A_step = A_sub * mask   # rows do not sum to 1

    # Calculate the initial alpha and final beta
    # defined inside of the one_step_em function to access gauss and A
    def forward_step(carry,x):
        # the forward_step function need to respect the scan function requirement
        old_alpha = carry

        # I need to repeat this for every k
        new_alpha = multivariate_normal.pdf(x,mean=mean[y_index],cov=covariance[y_index]) * (old_alpha@A_step)
        new_c = jnp.sum(new_alpha)
        new_alpha = new_alpha / new_c

        return new_alpha, (new_alpha, new_c)

    def backaward_step(carry,input):
        # the backward_step function need to respect the scan function requirement
        old_beta = carry
        x, c = input
        # I need to repeat this for every k
        new_beta = A_step @ (multivariate_normal.pdf(x,mean=mean[y_index],cov=covariance[y_index]) * old_beta)
        new_beta = new_beta / c

        return new_beta, new_beta

    def compute_step_csi(c,x,alpha,beta):
        csi_n = (jnp.outer(alpha,multivariate_normal.pdf(x,mean=mean[y_index],cov=covariance[y_index]) * beta) * A_step )/c
        return csi_n

    compute_csi = vmap(compute_step_csi,in_axes=(0,0,0,0))

    # Calculate the forward step
    _,output_forward = lax.scan(forward_step,initial_alpha,x[1:],reverse=False)
    alpha, c = output_forward
    alpha = jnp.vstack([initial_alpha[None, :], alpha])
    c = jnp.concatenate([jnp.array([c_0]),c])

    # Calculate the backward step
    _,beta = lax.scan(backaward_step,final_beta,(x[1:],c[1:]),reverse=True) # it is [1:] because the reverse starts form the last element
    beta = jnp.vstack([beta, final_beta[None, :]])

    # Calculate the parameters for our sentence
    # responsability
    responsability_step = alpha*beta
    # transition probability
    csi_step = compute_csi(c[1:],x[1:],alpha[:-1],beta[1:])

    # Adapt responsability and csi to the global case
    # for the responsability we want to go from (n,n) to (n,k)
    responsability_global = jnp.zeros((x.shape[0], num_gauss))
    responsability_global = responsability_global.at[:, y_index].add(responsability_step)

    # sum over every time
    csi_step_sum = jnp.sum(csi_step, axis=0)

    row_idx = y_index[:, None]  # shape (n, 1)
    col_idx = y_index[None, :]  # shape (1, n)

    # creating a (k,k) matrix
    csi_global_sum = jnp.zeros((num_gauss, num_gauss))
    csi_global_sum = csi_global_sum.at[row_idx, col_idx].add(csi_step_sum)

    # Return responsability and csi to be summed over every sequence
    return responsability_global, csi_global_sum
# END FUNCTION STEP_E

step_e_jit = jit(step_e)

# Usage of the function
def expectation_maximization(mean, covariance):

    responsability_sum = jnp.array([num_frame,num_gauss])
    responsability_x_sum = jnp.array([num_frame,num_gauss])
    csi_sum = jnp.array([num_gauss,num_gauss])
    covariance_sum = jnp.array([num_gauss,num_features,num_features])

    # Define a function that receive only two arguments and one output, as fori_loop requires
    @jit
    def body_fun(input):
        x, y, mean, covariance, responsability_sum, responsability_x_sum, csi_sum, covariance_sum = input
        responsability, csi = step_e_jit(x, y, mean, covariance)

        responsability_sum += responsability
        responsability_x_sum += (responsability*x)
        csi_sum += csi
        covariance_sum += jnp.sum(compute_covariance_jit(responsability,x,mean),0)

        return (responsability_sum, responsability_x_sum, csi_sum, covariance_sum)

    for (sentence,transcription) in next(dl.data_iter()):
        responsability_sum, responsability_x_sum, csi_sum, covariance_sum = body_fun(sentence,transcription,mean,covariance,responsability_x_sum, responsability_x_sum, csi_sum, covariance_sum)


    # Recalculate the gaussian distributions
    # 1. Recalculate the mean
    mean = responsability_x_sum/responsability_sum
    # 2. Recalculate the covariance
    covariance = covariance_sum/responsability_sum
    # Recalculate the A matrix
    A = csi_sum/jnp.sum(csi_sum,0) # rows must sum to 1

    return mean, covariance, A

# Call the function again
starting_time = timeit.default_timer()
mean, covariance, A = expectation_maximization(mean, covariance)
mean.block_until_ready()
final_time = timeit.default_timer()
execution_time = final_time - starting_time

print("Time for the execution", execution_time)
# Note: the execution time is way faster, on the first execution and also on the others

# PRINT THE RESULTS
