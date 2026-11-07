from jax import jit, numpy as jnp, lax
import numpy as jnp
from jax.scipy.stats import multivariate_normal
from scipy.special import softmax
import data_loader_speech as dl
from generate_chains import generate_chains
import project_1 as p1
import Levenshtein as lev
from itertools import islice

jnp.random.seed(42)

@jit
def viterbi(observations, transition_probs, initial_probs, μ, Σ):

    prev_step = initial_probs * multivariate_normal.pdf(observations[0],μ,Σ)

    # defined internally of another function so it can read the static arguments
    def viterbi_forward_step(carry,obs_t):
        transition = carry[:,None] * transition_probs # carry[:,None] is used to multiply the carry for every column of the transition probability table

        max_transition = jnp.max(transition, axis=0)
        best_j = jnp.argmax(transition, axis=0)

        new_carry = max_transition * multivariate_normal.pdf(obs_t,μ,Σ)

        return new_carry,best_j

    V,B = lax.scan(viterbi_forward_step, prev_step ,observations[1:], reverse=False) # [1:] because the first element was already used
    last_step = jnp.argmax(V)
    state_probs = jnp.max(V)

    def viterbi_backward_step(s_t,B_t):
        s_t_1 = B_t[s_t]
        return s_t_1, s_t_1

    _, map_seq = lax.scan(viterbi_backward_step,last_step,B,reverse=True)

    map_seq = jnp.append(map_seq, last_step)

    return map_seq, state_probs

@jit
def viterbi_log(observations, transition_probs, initial_probs, μ, Σ):

    prev_step = jnp.log(initial_probs) + multivariate_normal.logpdf(observations[0],μ,Σ)

    log_trans_prob = jnp.log(transition_probs)
    # defined internally of another function so it can read the static arguments
    def viterbi_forward_step_log(carry,obs_t):
        transition = carry[:,None] + log_trans_prob # carry[:,None] is used to multiply the carry for every column of the transition probability table

        max_transition = jnp.max(transition, axis=0)
        best_j = jnp.argmax(transition, axis=0)

        new_carry = max_transition + multivariate_normal.logpdf(obs_t,μ,Σ)

        return new_carry,best_j

    V,B = lax.scan(viterbi_forward_step_log, prev_step ,observations[1:], reverse=False) # [1:] because the first element was already used
    last_step = jnp.argmax(V)
    state_probs = jnp.max(V)

    def viterbi_backward_step(s_t,B_t):
        s_t_1 = B_t[s_t]
        return s_t_1, s_t_1

    _, map_seq = lax.scan(viterbi_backward_step,last_step,B,reverse=True)

    map_seq = jnp.append(map_seq, last_step)

    return map_seq, state_probs

# @partial(jit, static_argnums=(5,)) #used to indicate that the parameter h does not change during the programm
def h_best_viterbi_log(observations, transition_probs, initial_probs, μ, Σ, h):
    # I need to reshaper everything because if I ask for the 3 best paths and there are only 2 states this is not possible
    k = transition_probs.shape[0]
    v0 = jnp.log(initial_probs) + multivariate_normal.logpdf(observations[0],μ,Σ)
    prev_step = jnp.full((k,h), -jnp.inf) # at the beginning it exist only one best path
    prev_step = prev_step.at[:,0].set(v0) # insert v0 in the first column

    log_trans_prob = jnp.log(transition_probs)
    
    # defined internally of another function so it can read the static arguments
    @jit
    def viterbi_forward_step_log(carry,obs_t):
        transition = carry[:,:,None] + log_trans_prob[:,None,:] #it is used to have the same dimensions for the matrices
        scores_for_k = transition.reshape(k*h,k).T

        # top val contains the 3 best paths, top_indices contains their indices
        top_val, top_indices = lax.top_k(scores_for_k, h)

        best_j = top_indices // h # indicates from which state does every best indices come from
        best_r = top_indices % h  # indicates which type of path was it (first best, second best, ....)

        new_carry = top_val + multivariate_normal.logpdf(obs_t,μ,Σ)[:,None]

        return new_carry,(best_j,best_r)

    V,(B_j,B_r) = lax.scan(viterbi_forward_step_log, prev_step ,observations[1:], reverse=False) # [1:] because the first element was already used

    V = V.at[jnp.arange(k) != 41].set(-jnp.inf) # set to -inf all the states that are not 41, because we want to end in state 41


    best_log_probs, final_flat_index = lax.top_k(V.ravel(), h)
    last_states = final_flat_index // h
    last_ranks = final_flat_index % h

    @jit
    def viterbi_backward_step(carry,b_t):
        current_states, current_ranks = carry
        b_j_t, b_r_t = b_t

        prev_states = b_j_t[current_states, current_ranks]
        prev_ranks = b_r_t[current_states, current_ranks]

        return (prev_states,prev_ranks), prev_states

    _, map_seq = lax.scan(viterbi_backward_step,(last_states,last_ranks),(B_j,B_r),reverse=True)

    best_sequences = jnp.vstack([map_seq, last_states[None, :]]).T

    return best_sequences, best_log_probs


def collapse_duplicate(sequence):
    # This function collapses consecutive duplicate states in a sequence. For example, if the input sequence is [1, 1, 2, 2, 3], the output will be [1, 2, 3].
    sequence = jnp.asarray(sequence)
    if len(sequence) == 0:
        return sequence
    mask = jnp.ones(len(sequence), dtype=bool) # Create a boolean mask initialized to True for all elements
    mask[1:] = sequence[1:] != sequence[:-1] # Set mask to False for elements that are the same as the previous element
    return sequence[mask]

def main():
    BEST = 30
    N_TRAIN = 700 # we used 700 training samples to train the HMM
    START_STATE = 40 # Start is phoneme number 40
    
    params = jnp.load("trained_params.npz")
    mean = jnp.asarray(params["mean"], dtype=jnp.float32)
    cov = jnp.asarray(params["covariance"], dtype=jnp.float32)
    A = jnp.asarray(params["A"], dtype=jnp.float32)
    k, d = mean.shape
    
    init_probs = jnp.zeros(k).at[START_STATE].set(1.0)
    
    # iterate only on the test set, skipping the first N_TRAIN samples    
    test_iter = islice(dl.data_iter(), N_TRAIN, None)
    
    viterbi_errors = []
    random_errors = []

    for x, y in test_iter:
        y = jnp.asarray(y)

        # decode the 30-BEST
        seqs, lls = h_best_viterbi_log(jnp.asarray(x), A, init_probs, mean, cov, BEST)
        seqs, lls = jnp.asarray(seqs), jnp.asarray(lls)

        # discard sequences where there is no valid path from the start to the end
        if not jnp.isfinite(lls).any():
            continue

        # compute the weights for each sequence based on their log-likelihoods, and calculate the weighted average Levenshtein distance to the true sequence
        weights = softmax(jnp.where(jnp.isfinite(lls), lls, -jnp.inf))
        dists = [lev.distance(collapse_duplicate(s), y) for s in seqs]
        viterbi_errors.append(jnp.dot(weights, dists) / len(y))
        
        # generate a random sequence and compute the Levenshtein distance to the true sequence to prove that the Viterbi algorithm is better than random guessing
        chain = generate_chains(k,1,length=len(x),d=d,μ=jnp.asarray(mean),Σ=jnp.asarray(cov),transition_probs=A,initial_probs=init_probs)[0]
        random_errors.append(lev.distance(jnp.asarray(chain[2]), y) / len(y))

    viterbi_errors = jnp.array(viterbi_errors)
    random_errors = jnp.array(random_errors)

    # save informations in a npz file to be used in the report and printed with a different file
    jnp.savez("viterbi_results.npz",viterbi_errors=viterbi_errors,viterbi_mean_error=float(viterbi_errors.mean()),random_errors=random_errors,random_mean_error=float(random_errors.mean()),h=BEST,n_train=N_TRAIN)

if __name__ == "__main__":
    main()
