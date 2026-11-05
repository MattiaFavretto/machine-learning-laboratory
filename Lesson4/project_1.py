import jax
from jax import numpy as jnp, random as rnd, vmap, jit, lax
from jax.scipy.stats import multivariate_normal
from jax.scipy.linalg import solve_triangular
import pandas as pd
import timeit
### START CLAUDE
import os
import traceback
### END CLAUDE
import matplotlib.pyplot as plt
import data_loader_speech as dl
import generate_chains as gc
import numpy as np
from jax.scipy.special import logsumexp

# TODO - UNDERSTAND NEW LOGPDF VARIANT

### START CLAUDE
# Where results are written (set RESULTS_DIR in the job script, e.g. /share/$USER/project1)
RESULTS_DIR = os.environ.get("RESULTS_DIR", ".")
os.makedirs(RESULTS_DIR, exist_ok=True)
### END CLAUDE



### START CLAUDE
def prepare_padded_data(data_list):
    """Pad all (x, y) sequences to (max_T, D) and (max_n,) so @jit compiles only ONCE."""
    max_T = max(x.shape[0] for x, _ in data_list)
    max_n = max(y.shape[0] for _, y in data_list)
    X_pad = jnp.stack([jnp.pad(x, ((0, max_T - x.shape[0]), (0, 0))) for x, _ in data_list])
    Y_pad = jnp.stack([jnp.pad(y, (0, max_n - y.shape[0])) for _, y in data_list])
    T_lens = jnp.array([x.shape[0] for x, _ in data_list], dtype=jnp.int32)
    n_lens = jnp.array([y.shape[0] for _, y in data_list], dtype=jnp.int32)
    return X_pad, Y_pad, T_lens, n_lens


@jit
def compute_all_logpdf(X_pad, mean, covariance):
    """Compute multivariate normal logpdf across all padded sentences with a single Cholesky per phoneme."""
    K, D = mean.shape
    N, max_T, _ = X_pad.shape
    L = jnp.linalg.cholesky(covariance)  # (K, D, D)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)  # (K,)
    diff = X_pad[:, :, None, :] - mean[None, None, :, :]  # (N, max_T, K, D)
    z = solve_triangular(L, diff.transpose(2, 3, 0, 1).reshape(K, D, -1), lower=True)
    maha = jnp.sum(z ** 2, axis=1).T.reshape(N, max_T, K)
    return -0.5 * (D * jnp.log(2.0 * jnp.pi) + log_det[None, None, :] + maha)


@jit
def step_e_padded(pdf_full, y_pad, T_len, n_len, A):
    """JIT-compiled E-step on a single padded sentence (max_T, max_n) with masks for true lengths T_len, n_len."""
    max_T = pdf_full.shape[0]
    max_n = y_pad.shape[0]
    num_gauss = A.shape[0]

    # Select emission log-probabilities for the sentence transcription phonemes: (max_T, max_n)
    pdf = pdf_full[:, y_pad]
    n_idx = jnp.arange(max_n)
    valid_n = n_idx < n_len

    # Mask out padded phoneme positions (>= n_len)
    pdf = jnp.where(valid_n[None, :], pdf, -jnp.inf)

    # Initial state distribution pi = [1, 0, ..., 0] in log space
    pi = jnp.where(n_idx == 0, 0.0, -jnp.inf)
    initial_alpha = pi + pdf[0]
    c_0 = logsumexp(initial_alpha)
    c_0_safe = jnp.where(jnp.isfinite(c_0), c_0, 0.0)
    initial_alpha = initial_alpha - c_0_safe

    # Left-to-right transition matrix for the sentence states
    A_sub = A[y_pad[:, None], y_pad[None, :]]
    mask = jnp.eye(max_n) + jnp.eye(max_n, k=1)
    valid_pair = valid_n[:, None] & valid_n[None, :]
    A_step = jnp.where((mask > 0) & valid_pair, jnp.log(A_sub), -jnp.inf)

    def forward_step(old_alpha, inp):
        pdf_t, t_idx = inp
        new_alpha = pdf_t + logsumexp(old_alpha[:, None] + A_step, axis=0)
        new_c = logsumexp(new_alpha)
        new_c_safe = jnp.where(jnp.isfinite(new_c), new_c, 0.0)
        new_alpha = new_alpha - new_c_safe
        is_valid_t = t_idx < T_len
        alpha_out = jnp.where(is_valid_t, new_alpha, old_alpha)
        c_out = jnp.where(is_valid_t, new_c_safe, 0.0)
        return alpha_out, (alpha_out, c_out)

    t_indices = jnp.arange(1, max_T)
    _, (alpha_rest, c_rest) = lax.scan(forward_step, initial_alpha, (pdf[1:], t_indices))
    alpha = jnp.vstack([initial_alpha[None, :], alpha_rest])
    c = jnp.concatenate([jnp.array([c_0_safe]), c_rest])

    # Poiché la catena è vincolata a terminare nell'ultimo stato (n_len - 1),
    # impostiamo final_beta[n_len - 1] = -alpha[T_len - 1, n_len - 1] così che
    # sum_j gamma_t(j) = 1.0 e sum_{i,j} csi_t(i,j) = 1.0 (vere PMF) senza underflow!
    last_alpha_end = alpha[-1, n_len - 1]
    last_alpha_safe = jnp.where(jnp.isfinite(last_alpha_end), last_alpha_end, 0.0)
    final_beta = jnp.where(n_idx == (n_len - 1), -last_alpha_safe, -jnp.inf)

    def backward_step(old_beta, inp):
        pdf_t, c_t, t_idx = inp
        new_beta = logsumexp(A_step + (pdf_t + old_beta)[None, :], axis=1) - c_t
        new_beta = jnp.where(valid_n, new_beta, -jnp.inf)
        is_valid_t = t_idx < T_len
        beta_out = jnp.where(is_valid_t, new_beta, old_beta)
        return beta_out, beta_out

    _, beta_rest = lax.scan(backward_step, final_beta, (pdf[1:], c[1:], t_indices), reverse=True)
    beta = jnp.vstack([beta_rest, final_beta[None, :]])

    valid_t = jnp.arange(max_T) < T_len
    responsability_step = jnp.where(valid_t[:, None] & valid_n[None, :], jnp.exp(alpha + beta), 0.0)

    def compute_step_csi(c_t, pdf_t, alpha_t, beta_t, t_idx):
        csi_n = jnp.exp(alpha_t[:, None] + A_step + (pdf_t + beta_t)[None, :] - c_t)
        return jnp.where((t_idx < T_len) & valid_pair, csi_n, 0.0)

    csi_step = vmap(compute_step_csi)(c[1:], pdf[1:], alpha[:-1], beta[1:], t_indices)
    csi_step_sum = jnp.sum(csi_step, axis=0)

    responsability_global = jnp.zeros((max_T, num_gauss)).at[:, y_pad].add(responsability_step)
    csi_global_sum = jnp.zeros((num_gauss, num_gauss)).at[y_pad[:, None], y_pad[None, :]].add(csi_step_sum)
    ll = jnp.sum(c) + last_alpha_safe
    return responsability_global, csi_global_sum, ll


@jit
def em_epoch_padded(i, mean, covariance, A, total_ll_array, X_pad, Y_pad, T_lens, n_lens):
    """Single EM epoch compiled with @jit and vectorized across all padded sentences."""
    num_gauss, num_features = mean.shape
    all_pdfs = compute_all_logpdf(X_pad, mean, covariance)
    resps, csis, lls = vmap(step_e_padded, in_axes=(0, 0, 0, 0, None))(all_pdfs, Y_pad, T_lens, n_lens, A)

    # Ignora eventuali frasi patologiche in cui la sequenza finale non è raggiungibile (es. T_len < n_len)
    valid_sent = jnp.isfinite(lls) & (~jnp.isnan(resps).any(axis=(1, 2))) & (~jnp.isnan(csis).any(axis=(1, 2)))
    resps = jnp.where(valid_sent[:, None, None], resps, 0.0)
    csis = jnp.where(valid_sent[:, None, None], csis, 0.0)
    lls = jnp.where(valid_sent, lls, 0.0)

    Nk_raw = jnp.sum(resps, axis=(0, 1))
    Nk = jnp.maximum(Nk_raw, 1e-8)
    Sx = jnp.einsum('ntk,ntd->kd', resps, X_pad)
    # Se un fonema non compare mai nel training set (es. START/END), mantieni la media precedente
    new_mean = jnp.where((Nk_raw > 1e-3)[:, None], Sx / Nk[:, None], mean)

    # Formula centrata E[(x - mu)(x - mu)^T]: garantisce semidefinitezza positiva senza cancellazione numerica in float32
    diff = X_pad[:, :, None, :] - new_mean[None, None, :, :]
    new_cov = jnp.einsum('ntk,ntkd,ntke->kde', resps, diff, diff) / Nk[:, None, None]
    new_cov = jnp.where((Nk_raw > 1e-3)[:, None, None], new_cov, covariance)
    new_cov = new_cov + 1e-3 * jnp.eye(num_features)[None, :, :]

    csi_sum = jnp.sum(csis, axis=0)
    row_sum = jnp.sum(csi_sum, axis=1, keepdims=True)
    # Se un fonema non ha transizioni uscenti (row_sum == 0), evita 0/0 = NaN mantenendo una riga uniforme
    new_A = jnp.where(row_sum > 1e-8, csi_sum / jnp.maximum(row_sum, 1e-8), jnp.ones_like(A) / num_gauss)
    new_A = new_A + 1e-8 * jnp.eye(num_gauss)
    new_A = new_A / jnp.sum(new_A, axis=1, keepdims=True)

    total_ll = jnp.sum(lls)
    total_ll_array = total_ll_array.at[i].set(total_ll)
    return new_mean, new_cov, new_A, total_ll_array
### END CLAUDE


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

    # Poiche la catena termina nello stato n-1, final_beta[-1] = -alpha[-1, -1]
    # rende gamma e csi delle vere PMF (somma = 1 a ogni istante t) ed evita l'underflow!
    final_beta = jnp.full(n, -jnp.inf).at[-1].set(-alpha[-1, -1])

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
    ll = jnp.sum(c) + alpha[-1, -1]

    # Return responsability and csi to be summed over every sequence
    return responsability_global, csi_global_sum, ll
# END FUNCTION STEP_E

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
    # try_with_generate_chains()

    try:
        # Load phonemes list
        _, phonemes_list = dl.get_phonemes()
        phoneme2idx = {p: i for i, p in enumerate(phonemes_list)} # I convert phonems to index because I need that to use the evidence in the e_step
        num_gauss = len(phonemes_list) # there is a gaussian for every phoneme

        # 2. Upload the sentences and transcriptions and split them into training and test set, translating the phonemes into indexes
        train_data = []
        test_data = []

        print("Loading first 700 training sentences from dataset...", flush=True)
        for i, (sentence, transcription) in enumerate(dl.data_iter()):
            if i >= 700:
                # Non carichiamo le restanti ~12.400 frasi .wav durante il training del punto c)
                break
            x = jnp.array(sentence)
            # data_iter() returns transcription already converted to phoneme indices!
            y_index = jnp.array(transcription, dtype=jnp.int32)
            train_data.append((x, y_index))
            if (i + 1) % 100 == 0:
                print(f"Loaded {i + 1}/700 sentences...", flush=True)

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

        # Padding di tutte le 700 frasi a (max_T, num_features) e (max_n,) per compilare step_e con @jit una sola volta!
        X_pad, Y_pad, T_lens, n_lens = prepare_padded_data(train_data)
        print(f"Padded dataset shapes: X_pad={X_pad.shape}, Y_pad={Y_pad.shape}", flush=True)

        starting_time = timeit.default_timer()
        total_ll_array = jnp.zeros(epochs)

        print(f"Starting training ({epochs} epochs on {len(train_data)} sentences with lax.fori_loop + @jit)...", flush=True)

        def epoch_body(i, carry):
            m, cov, trans_A, ll_arr = carry
            m, cov, trans_A, ll_arr = em_epoch_padded(i, m, cov, trans_A, ll_arr, X_pad, Y_pad, T_lens, n_lens)
            jax.debug.print("Epoch {}/{} completed - LL: {}", i + 1, epochs, ll_arr[i])
            return (m, cov, trans_A, ll_arr)

        mean, covariance, A, total_ll_array = lax.fori_loop(
            0, epochs, epoch_body, (mean, covariance, A, total_ll_array)
        )
        mean.block_until_ready()

        final_time = timeit.default_timer()
        execution_time = final_time - starting_time

        print("Time for the execution:", execution_time, flush=True)

        # PRINT THE RESULTS
        print("Final mean", mean)
        print("Final covariance", covariance)
        print("Final A", A)

        ### START CLAUDE
        # Save the trained parameters and the LL per epoch (plot it later, locally, from trained_params.npz)
        np.savez(os.path.join(RESULTS_DIR, "trained_params.npz"),
                 mean=np.array(mean), covariance=np.array(covariance), A=np.array(A),
                 ll=np.array(total_ll_array))
        ### END CLAUDE

    except Exception as e:
        print("An error occured", e)
        ### START CLAUDE
        traceback.print_exc()  # show where the error happened instead of hiding it
        ### END CLAUDE


if __name__ == "__main__":
    main()
