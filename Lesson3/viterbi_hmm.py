from jax import jit, numpy as jnp, vmap, lax
import numpy as np
from generate_chains import generate_chains
from jax.scipy.stats import multivariate_normal
import timeit
from functools import partial

np.random.seed(42)

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

@partial(jit, static_argnums=(5,)) #used to indicate that the parameter h does not change during the programm
def h_best_viterbi_log(observations, transition_probs, initial_probs, μ, Σ, h):
    # I need to reshaper everything because if I ask for the 3 best paths and there are only 2 states this is not possible
    k = transition_probs.shape[0]
    v0 = jnp.log(initial_probs) + multivariate_normal.logpdf(observations[0],μ,Σ)
    prev_step = jnp.full((k,h), -jnp.inf) # at the beginning it exist only one best path
    prev_step = prev_step.at[:,0].set(v0) # insert v0 in the first column

    log_trans_prob = jnp.log(transition_probs)
    # defined internally of another function so it can read the static arguments
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


    best_log_probs, final_flat_index = lax.top_k(V.ravel(), h)
    last_states = final_flat_index // h
    last_ranks = final_flat_index % h

    def viterbi_backward_step(carry,b_t):
        current_states, current_ranks = carry
        b_j_t, b_r_t = b_t

        prev_states = b_j_t[current_states, current_ranks]
        prev_ranks = b_r_t[current_states, current_ranks]

        return (prev_states,prev_ranks), prev_states

    _, map_seq = lax.scan(viterbi_backward_step,(last_states,last_ranks),(B_j,B_r),reverse=True)

    best_sequences = jnp.vstack([map_seq, last_states[None, :]]).T

    return best_sequences, best_log_probs


def main():
    ## Data generation
    states = ["A", "B", "C", "D", "E", "F"]
    k = len(states)
    d = 3
    h = 3
    transition_probs = np.random.uniform(0, 1, size=(k,) * 2) + np.eye(k) * 2 # used to increase the probability of remaining in the same state
    transition_probs = (
        transition_probs / np.sum(transition_probs, axis=0)
    ).T  # row stochastic
    initial_probs = np.array(
        [
            1 / k,
        ]
        * k
    )
    μ = np.random.normal(size=(k, d))
    Σ_sqrt = np.random.normal(size=(k, d, d))
    Σ = Σ_sqrt.transpose((0, 2, 1)) @ Σ_sqrt

    chains = generate_chains(
        k,
        100,
        length=100,
        d=d,
        transition_probs=transition_probs,
        initial_probs=np.array([1 / k] * k),
        μ=μ,
        Σ=Σ,
    )

    for length in [10, 20, 100]:
            print("------------------- TESTING CHAIN LENGTH:",length, "---------------------------------------")
            chains = generate_chains(
                k,
                100,
                length=length,
                d=d,
                transition_probs=transition_probs,
                initial_probs=initial_probs,
                μ=μ,
                Σ=Σ,
            )

            start = timeit.default_timer()
            for true_states, observations, _ in chains:
                map_seq, probabilities = viterbi(observations, transition_probs, initial_probs, μ, Σ)
                print("BEST PATH WITH VITERBI ALGORITHM")
                print(true_states)
                print(map_seq, probabilities)
            end = timeit.default_timer()
            print("Time of execution classical, length: ", length, " : ", end - start)

            start = timeit.default_timer()
            for true_states, observations, _ in chains:
                map_seq, probabilities = viterbi_log(observations, transition_probs, initial_probs, μ, Σ)
                print("BEST PATH WITH LOGARITHMIC VITERBI ALGORITHM")
                print(true_states)
                print(map_seq, probabilities)
            end = timeit.default_timer()
            print("Time of execution logaritmic, length: ", length, " : ", end - start)
            start = timeit.default_timer()
            for true_states, observations, _ in chains:
                top_seq, top_prob = h_best_viterbi_log(observations, transition_probs, initial_probs, μ, Σ, h)
                print(h, "BEST PATHS WITH LOGARITHMIC VITERBI ALGORITHM")
                print(true_states)
                print(top_seq, top_prob)
            end = timeit.default_timer()
            print("Time of execution h-best logaritmic, length: ", length, " : ", end - start)

if __name__ == "__main__":
    main()
