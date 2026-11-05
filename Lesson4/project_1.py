from jax import numpy as jnp, random as rnd, vmap, jit, lax
from jax.scipy.stats import multivariate_normal
import pandas as pd
import timeit
import matplotlib.pyplot as plt
import data_loader_speech as dl
import generate_chains as gc
import numpy as np
from jax.scipy.special import logsumexp

# TODO - UNDERSTAND NEW LOGPDF VARIANT



# input: x are the evidence, y are the hidden states, mean and covariance are the parameters of the gaussians, A is the transition matrix
@jit
def step_e(x,y,mean,covariance, A):
    # Get di length (n) of the sentence and map the indeces of sentence phonemes considering the global list of phonems
    n = len(y)
    num_gauss = mean.shape[0]

    # Calculate the multivariate logpdf, the pdf gives 0.0 
    pdf = multivariate_normal.logpdf(
        x[:, None, :], mean=mean[y][None, :, :], cov=covariance[y][None, :, :, :]
    )

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
    A_step = jnp.log(A_step) # move to log space

    # Calculate the initial alpha and final beta
    # defined inside of the one_step_em function to access gauss and A
    def forward_step(carry,pdf_t):
        # the forward_step function need to respect the scan function requirement
        old_alpha = carry

        # I need to repeat this for every k
        new_alpha = pdf_t + logsumexp(old_alpha[:, None] + A_step, axis=0)
        new_c = logsumexp(new_alpha)
        new_alpha -= new_c

        return new_alpha, (new_alpha, new_c)

    def backaward_step(carry,input):
        # the backward_step function need to respect the scan function requirement
        old_beta = carry
        pdf_t, c_t = input
        # I need to repeat this for every k
        new_beta = logsumexp(A_step + (pdf_t + old_beta)[None, :], axis=1) - c_t

        return new_beta, new_beta

    def compute_step_csi(c_t,pdf_t,alpha_t,beta_t):
        csi_n = jnp.exp(alpha_t[:, None] + A_step + (pdf_t + beta_t)[None, :] - c_t)
        return csi_n

    compute_csi = vmap(compute_step_csi,in_axes=(0,0,0,0))

    # Calculate the forward step
    _,output_forward = lax.scan(forward_step,initial_alpha,pdf[1:],reverse=False)
    alpha, c = output_forward
    alpha = jnp.vstack([initial_alpha[None, :], alpha])
    c = jnp.concatenate([jnp.array([c_0]),c])

    # Calculate the backward step
    _,beta = lax.scan(backaward_step,final_beta,(pdf[1:],c[1:]),reverse=True) # it is [1:] because the reverse starts form the last element
    beta = jnp.vstack([beta, final_beta[None, :]])

    # Calculate the parameters for our sentence
    # responsability
    responsability_step = jnp.exp(alpha + beta)
    # transition probability
    csi_step = compute_csi(c[1:],pdf[1:],alpha[:-1],beta[1:])

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
    ll = jnp.sum(c)

    # Return responsability and csi to be summed over every sequence
    return responsability_global, csi_global_sum, ll
# END FUNCTION STEP_E

@jit
def update_stats(x, y, mean, covariance, A, Nk, Sx, Sxx, csi_sum, total_ll):
    responsability, csi, ll = step_e(x, y, mean, covariance, A)
    Nk += jnp.sum(responsability, axis=0)
    Sx += jnp.einsum('tk,td->kd', responsability, x)
    Sxx += jnp.einsum('tk,td,te->kde', responsability, x, x)
    csi_sum += csi
    total_ll += ll
    return Nk, Sx, Sxx, csi_sum, total_ll

def expectation_maximization(i, mean, covariance, A, total_ll_array, num_gauss, num_features, train_data):

    csi_sum = jnp.zeros((num_gauss, num_gauss))
    # If I want to compute the covariance I would need to compute first the new mean and then compute the covariance.
    Nk = jnp.zeros((num_gauss,))
    Nk += 1e-8 # add a small value to avoid division by zero
    Sx = jnp.zeros((num_gauss, num_features))
    Sxx = jnp.zeros((num_gauss, num_features, num_features))
    total_ll = 0.0

    for (x,y) in train_data:
        Nk, Sx, Sxx, csi_sum, total_ll = update_stats(x, y, mean, covariance, A, Nk, Sx, Sxx, csi_sum, total_ll)

    # Recalculate the gaussian distributions
    # 1. Recalculate the mean
    mean = Sx/Nk[:, None]
    # 2. Recalculate the covariance
    covariance = Sxx/Nk[:, None, None] - jnp.einsum('kd,ke->kde', mean, mean)
    covariance += 1e-5 * jnp.eye(num_features)[None, :, :] 
    # Recalculate the A matrix
    A = csi_sum/jnp.sum(csi_sum,1)[:,None] # rows must sum to 1
    A += 1e-8 * jnp.eye(num_gauss) # add a small value to the diagonal to avoid zero probabilities


    total_ll_array = total_ll_array.at[i].set(total_ll)
    return mean, covariance, A, total_ll_array


def try_with_generate_chains():
    print("Generate chains started")
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
    seed = 1701
    k1 = rnd.key(seed)

    all_frames = jnp.vstack([x for x, _ in train])
    idx = rnd.choice(k1, all_frames.shape[0], shape=(num_gauss,), replace=False)
    mean = all_frames[idx]
    covariance = jnp.tile(jnp.cov(all_frames, rowvar=False)[None,:,:],(num_gauss,1,1)) # used to repeat the covariance for every gaussian.

    total_ll_array = jnp.zeros(epochs)
    A = jnp.ones((num_gauss,num_gauss))/num_gauss
    ### START CLAUDE
    mse_mean = []
    mse_cov = []
    mse_A = []
    ### END CLAUDE
    for i in range(epochs):
            mean, covariance, A, total_ll_array = expectation_maximization(i,mean, covariance, A, total_ll_array, num_gauss, num_features, train)
            ### START CLAUDE
            # MSE between the estimated parameters and the true ones used to generate the chains
            mse_mean.append(float(np.mean((np.array(mean) - data_mean) ** 2)))
            mse_cov.append(float(np.mean((np.array(covariance) - data_cov) ** 2)))
            mse_A.append(float(np.mean((np.array(A) - data_A) ** 2)))
            print(f"Epoch {i+1}: MSE mean={mse_mean[-1]:.5f}, cov={mse_cov[-1]:.5f}, A={mse_A[-1]:.5f}")
            ### END CLAUDE

    print("Final mean", mean, "Initial mean", data_mean)
    print("Final covariance", covariance, "Initial covariance", data_cov)
    print("Final A", A, "Initial A", data_A)

    ### START CLAUDE
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
    ### END CLAUDE



def main():

    # Test for point b) of the project
    try_with_generate_chains()

    try:
        # Load phonemes list
        _, phonemes_list = dl.get_phonemes()
        phoneme2idx = {p: i for i, p in enumerate(phonemes_list)} # I convert phonems to index because I need that to use the evidence in the e_step
        num_gauss = len(phonemes_list) # there is a gaussian for every phoneme

        # 2. Upload the sentences and transcriptions and split them into training and test set, translating the phonemes into indexes
        train_data = []
        test_data = []

        for i, (sentence, transcription) in enumerate(dl.data_iter()):
            x = jnp.array(sentence)
            y_index = jnp.array([phoneme2idx[p] for p in transcription], dtype=jnp.int32)

            if i < 700:
                train_data.append((x, y_index))
            else:
                test_data.append((x, y_index))

        num_features = train_data[0][0].shape[1] # the number of features is the dimension of the input
        # parameters of the gaussians

        # create random values for the two gaussian
        seed = 1701
        k1 = rnd.key(seed) # unless they will have the same value

        # Computer the initial mean and covariance for the gaussians, using all the sentences in the training set.
        all_train_frames = jnp.vstack([x for x, _ in train_data])
        num_frames = all_train_frames.shape[0]
        idx = rnd.choice(k1, num_frames, shape=(num_gauss,), replace=True)
        mean = all_train_frames[idx]

        covariance = jnp.tile(jnp.cov(all_train_frames, rowvar=False)[None,:,:],(num_gauss,1,1)) # used to repeat the covariance for every gaussian.

        epochs = 10
        A = jnp.ones((num_gauss,num_gauss))/num_gauss

        # Call the function again
        starting_time = timeit.default_timer()
        total_ll_array = jnp.zeros(epochs)

        for i in range(epochs):
            mean, covariance, A, total_ll_array = expectation_maximization(i,mean, covariance, A, total_ll_array, num_gauss, num_features, train_data)
        mean.block_until_ready()
        final_time = timeit.default_timer()
        execution_time = final_time - starting_time

        print("Time for the execution", execution_time)

        # PRINT THE RESULTS
        print("Final mean", mean)
        print("Final covariance", covariance)
        print("Final A", A)

    except Exception as e:
        print("An error occured", e)


if __name__ == "__main__":
    main()
