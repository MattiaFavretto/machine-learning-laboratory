from jax import numpy as jnp, random as rnd, vmap, jit, lax
from jax.scipy.stats import multivariate_normal
import os
import traceback
import matplotlib.pyplot as plt
import data_loader_speech as dl
import generate_chains as gc
import numpy as np
from jax.scipy.special import logsumexp

# Used to save the trained parameters and results
RESULTS_DIR = os.environ.get("RESULTS_DIR", ".")
os.makedirs(RESULTS_DIR, exist_ok=True)

def pad_data(data_list):
    
    max_T = max(x.shape[0] for x, _ in data_list) # calculate the maximum length of the sentences in the training set
    max_n = max(y.shape[0] for _, y in data_list) # calculate the maximum length of the transcriptions in the training set

    # Pad the sentences and transcriptions to the maximum length
    X_pad = jnp.stack([jnp.pad(x, ((0, max_T - x.shape[0]), (0, 0))) for x, _ in data_list])
    Y_pad = jnp.stack([jnp.pad(y, (0, max_n - y.shape[0])) for _, y in data_list])

    # Calculate the true lengths of the sentences and transcriptions, it will be used to mask the padded values in the E-step
    T_true = jnp.array([x.shape[0] for x, _ in data_list])
    n_true = jnp.array([y.shape[0] for _, y in data_list])
    return X_pad, Y_pad, T_true, n_true

@jit
def logpdf(x, mean, covariance):
    pdf_i = multivariate_normal.logpdf(x[:,None,:], mean, covariance)
    return pdf_i

logpdf_vmap = vmap(logpdf, in_axes=(0, None, None))  # calculate for all the sentences at the same time

# input: x are the evidence, y are the hidden states, mean and covariance are the parameters of the gaussians, A is the transition matrix
@jit
def step_e(pdf, x, y, T_true, n_true, A):
    # Get di length (n) of the sentence and map the indeces of sentence phonemes considering the global list of phonems
    n = len(y)
    num_gauss = A.shape[0]
    # Create a boolean mask to only consider the valid phonemes
    valid_n = jnp.arange(n) < n_true # jnp.arange is used to create the sequence of indices and then compare it with the true length of the sentence to create a boolean mask that is True for the valid phonemes and False for the padded ones.
    valid_t = jnp.arange(x.shape[0]) < T_true # boolean mask for only real timespamps, not the padded ones 

    pdf = pdf[:,y] # select only the pdf of the valid phonemes, so that we can use it in the forward and backward steps
    pdf = jnp.where(valid_n, pdf, -jnp.inf) # set the pdf of the padded phonemes to -inf, so that they will not contribute to the forward and backward steps

    # Initialize alfa and beta inside the function to deal also with sentences with different length
    pi = jnp.zeros(n).at[0].set(1.0)
    pi = jnp.log(pi) # move to log space
    initial_alpha = pi + pdf[0]
    c_0 = logsumexp(initial_alpha)
    initial_alpha -= c_0

    final_beta = jnp.full(n, -jnp.inf).at[-1].set(0.0)

    # Create the A matrix for this situation A_sub[i, j] = A[p(i), p(j)]
    A_sub = A[y[:, None], y[None, :]]

    # maske to only have left-to-right
    mask = jnp.eye(n) + jnp.eye(n, k=1)
    A_step = A_sub * mask   # rows do not sum to 1
    valid_pairs = valid_n[:, None] & valid_n[None, :]
    A_step = jnp.where((mask>0) & valid_pairs, jnp.log(A_step), -jnp.inf) # move to log space, I also don't want to make a transition on the padded values

    # Calculate the initial alpha and final beta
    # defined inside of the one_step_em function to access gauss and A
    def forward_step(carry,inp):
        # the forward_step function need to respect the scan function requirement
        old_alpha = carry
        pdf_t, t_sentence = inp

        # I need to repeat this for every k
        new_alpha = pdf_t + logsumexp(old_alpha[:, None] + A_step, axis=0)
        new_c = logsumexp(new_alpha)
        new_alpha -= new_c

        # If we are in the padding part of the sentence we don't want to update the alpha and c, so we keep the old values
        new_alpha = jnp.where(t_sentence < T_true, new_alpha, old_alpha)
        new_c = jnp.where(t_sentence < T_true, new_c, 0.0)

        return new_alpha, (new_alpha, new_c)

    def backaward_step(carry,input):
        # the backward_step function need to respect the scan function requirement
        old_beta = carry
        pdf_t, c_t, t_sentence = input
        # I need to repeat this for every k
        new_beta = logsumexp(A_step + (pdf_t + old_beta)[None, :], axis=1) - c_t
        new_beta = jnp.where(valid_n, new_beta, -jnp.inf)

        # do not update the beta if we are in the padding part of the sentence, so we keep the old values
        new_beta = jnp.where(t_sentence < T_true, new_beta, old_beta)

        return new_beta, new_beta

    def compute_step_csi(c_t,pdf_t,alpha_t,beta_t):
        csi_n = jnp.exp(alpha_t[:, None] + A_step + (pdf_t + beta_t)[None, :] - c_t)
        return csi_n

    compute_csi = vmap(compute_step_csi,in_axes=(0,0,0,0))

    # Calculate the forward step
    t_indices = jnp.arange(1, x.shape[0]) # the first element is already calculated, so we start from 1

    _,output_forward = lax.scan(forward_step,initial_alpha,(pdf[1:],t_indices),reverse=False)
    alpha, c = output_forward
    alpha = jnp.vstack([initial_alpha[None, :], alpha])
    c = jnp.concatenate([jnp.array([c_0]),c])

    # Poiche la catena termina nello stato n-1, final_beta[-1] = -alpha[-1, -1]
    # rende gamma e csi delle vere PMF (somma = 1 a ogni istante t) ed evita l'underflow!
    final_beta = jnp.full(n, -jnp.inf).at[n_true - 1].set(-alpha[-1, n_true -1])

    # Calculate the backward step
    _,beta = lax.scan(backaward_step,final_beta,(pdf[1:],c[1:], t_indices),reverse=True) # it is [1:] because the reverse starts form the last element
    beta = jnp.vstack([beta, final_beta[None, :]])

    # Create a temporal mask to only consider the valid time steps


    # Calculate the parameters for our sentence
    # responsability
    responsability_step = jnp.where(valid_t[:, None] & valid_n[None, :], jnp.exp(alpha + beta), 0.0)
    # transition probability
    csi_step = compute_csi(c[1:],pdf[1:],alpha[:-1],beta[1:])
    csi_step = jnp.where(valid_t[1:, None, None] & valid_pairs[None, :, :], csi_step, 0.0)

    # Adapt responsability and csi to the global case
    # for the responsability we want to go from (n,n) to (n,k)
    responsability_global = jnp.zeros((x.shape[0], num_gauss))
    responsability_global = responsability_global.at[:, y].add(responsability_step)

    # sum over every time
    csi_step_sum = jnp.sum(csi_step, axis=0)

    row_idx = y[:, None]  # shape (n, 1)
    col_idx = y[None, :]  # shape (1, n)

    # creating a (k,k) matrix
    csi_global_sum = jnp.zeros((num_gauss, num_gauss))
    csi_global_sum = csi_global_sum.at[row_idx, col_idx].add(csi_step_sum)

    # Calculation of the Log Likelihood
    ll = jnp.sum(c) + alpha[-1, n_true -1]

    # Return responsability and csi to be summed over every sequence
    return responsability_global, csi_global_sum, ll
# END FUNCTION STEP_E

@jit
def expectation_maximization(i, mean, covariance, A, total_ll_array, x, y, T_true, n_true):
    num_gauss, num_features = mean.shape

    pdf = logpdf_vmap(x, mean, covariance)

    responsability, csi, ll = vmap(step_e, in_axes=(0,0,0,0,0,None))(pdf,x,y,T_true,n_true,A)

    csi_sum = jnp.zeros((num_gauss, num_gauss))
    # If I want to compute the covariance I would need to compute first the new mean and then compute the covariance.
    Nk = jnp.sum(responsability, axis=(0,1)) + 1e-8 # add a small value to avoid division by zero
    Sx = jnp.einsum('ntk,ntd->kd', responsability, x)
    Sxx = jnp.einsum('ntk,ntd,nte->kde', responsability, x,x)
    csi_sum = jnp.sum(csi,axis=0)
    total_ll = jnp.sum(ll)

    # Recalculate the gaussian distributions
    # 1. Recalculate the mean
    mean = Sx/Nk[:, None]
    # 2. Recalculate the covariance
    covariance = Sxx/Nk[:, None, None] - jnp.einsum('kd,ke->kde', mean, mean)
    covariance += 1e-5 * jnp.eye(num_features)[None, :, :] 
    # Recalculate the A matrix
    A = csi_sum/jnp.sum(csi_sum,1)[:,None] + 1e-8 # rows must sum to 1, add also a term to avoid 0/0
    A += 1e-8 * jnp.eye(num_gauss) # add a small value to the diagonal to avoid zero probabilities
    A = A/jnp.sum(A,axis=1)[:,None]

    return mean, covariance, A, total_ll_array


def try_with_generate_chains(k1):
    # This function is used not on the cluster, to test the EM algorithm with generated chains, to see if it converges to the true parameters used to generate the chains.
    # At the end it plots the data withouth using another file 

    # Set the parameters for the generated chains
    num_gauss = 4
    num_features = 3
    epochs = 10
    num_frames = 50
    frame_length = 30

    # Generate random parameters for the gaussians
    data_mean = np.random.normal(size=(num_gauss, num_features))

    sqrt_cov = np.random.normal(size=(num_gauss, num_features, num_features))
    data_cov = sqrt_cov.transpose((0, 2, 1)) @ sqrt_cov

    data_A = np.random.uniform(0.1, 1.0, size=(num_gauss, num_gauss)) + np.eye(num_gauss) * 2.0
    data_A = data_A / np.sum(data_A, axis=1, keepdims=True)

    # Generate chains based on the random parameters
    chains = gc.generate_chains(states=num_gauss, amount=num_frames, length=frame_length, d=num_features, μ=data_mean,
    Σ=data_cov, transition_probs=data_A) # generate 50 chains of length 30 with 4 states and 3 features

    train = [(jnp.array(c[1]), jnp.array(c[2], dtype=jnp.int32)) for c in chains]

    # Initialize parameters for the EM algorithm
    all_frames = jnp.vstack([x for x, _ in train])
    idx = rnd.choice(k1, all_frames.shape[0], shape=(num_gauss,), replace=False)
    mean = all_frames[idx]
    covariance = jnp.tile(jnp.cov(all_frames, rowvar=False)[None,:,:],(num_gauss,1,1)) # used to repeat the covariance for every gaussian.

    total_ll_array = jnp.zeros(epochs)
    A = jnp.ones((num_gauss,num_gauss))/num_gauss
    
    # Creating the lists to store the MSE for mean, covariance and A, to calculate it at the end of every epoch
    mse_mean = []
    mse_cov = []
    mse_A = []

    x_pad, y_pad, T_true, n_true = pad_data(train)

    for i in range(epochs):
            mean, covariance, A, total_ll_array = expectation_maximization(i,mean, covariance, A, total_ll_array, x_pad,y_pad,T_true,n_true)
            
            # Calculating the MSE, to understand the difference between the estimated parameters and the real ones used to create the data
            mse_mean.append(float(np.mean((np.array(mean) - data_mean) ** 2)))
            mse_cov.append(float(np.mean((np.array(covariance) - data_cov) ** 2)))
            mse_A.append(float(np.mean((np.array(A) - data_A) ** 2)))

    # Plotting the MSE for mean, covariance and A over the epochs
    epochs_axis = np.arange(1, epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, mse_mean, marker="o", label="Mean")
    plt.plot(epochs_axis, mse_cov, marker="s", label="Covariance")
    plt.plot(epochs_axis, mse_A, marker="^", label="A")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("MSE vs true parameters per epoch")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()



def main():
    # create random values for the two gaussian
    seed = 1701
    k1 = rnd.key(seed) # unless they will have the same value
    epochs = 10
    num_training_sentences = 700

    # Test for point b) of the project
    # try_with_generate_chains(k1)

    try:
        # Load phonemes list and IPA transcriptions
        _, phonemes_list = dl.get_phonemes()
        num_gauss = len(phonemes_list) # there is a gaussian for every phoneme

        phonemes2ipa = dl.get_phonemes2ipa()
        phonemes_ipa = [phonemes2ipa([p]) for p in phonemes_list]
        
        data = []

        for i, (sentence, transcription) in enumerate(dl.data_iter()):
            if i >= num_training_sentences: 
                break
            x = jnp.array(sentence)
            y = jnp.array(transcription, dtype=jnp.int32)
            
            data.append((x, y))

        # Computer the initial mean and covariance for the gaussians, using all the sentences in the training set.
        all_sentences = jnp.vstack([x for x, _ in data])
        num_frames = all_sentences.shape[0]
        idx = rnd.choice(k1, num_frames, shape=(num_gauss,), replace=True)
        mean = all_sentences[idx]

        covariance = jnp.tile(jnp.cov(all_sentences, rowvar=False)[None,:,:],(num_gauss,1,1)) # used to repeat the covariance for every gaussian.

        A = jnp.ones((num_gauss,num_gauss))/num_gauss

        # In order to use the @jit decorator to the em fuction we need the sentences to have the same length, so we need to add a padding to sentences and transcriptions. 
        
        x, y, T_true, n_true = pad_data(data)

        total_ll_array = jnp.zeros(epochs)

        def epoch(i, carry):
            m, cov, trans_A, ll_arr = carry
            m, cov, trans_A, ll_arr = expectation_maximization(i, m, cov, trans_A, ll_arr, x, y, T_true, n_true)
            return (m, cov, trans_A, ll_arr)

        mean, covariance, A, total_ll_array = lax.fori_loop(
            0, epochs, epoch, (mean, covariance, A, total_ll_array)
        )
        mean.block_until_ready()

        # Save the trained parameters, LL and phonemes in a file, so can be used later to plot the LL and to use them in the Viterbi algorithm
        
        save_path = os.path.join(RESULTS_DIR, "trained_params.npz")
        np.savez(
            save_path,
            mean=np.array(mean),
            covariance=np.array(covariance),
            A=np.array(A),
            ll=np.array(total_ll_array),
            phonemes=np.array(phonemes_list),
            phonemes_ipa=np.array(phonemes_ipa),
        )

    except Exception as e:
        print("An error occured", e)
        traceback.print_exc()  # show the error 

if __name__ == "__main__":
    main()
