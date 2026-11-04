import numpy as np
from jax import random, jit, numpy as jnp
from jax.lax import scan, dynamic_index_in_dim
from scipy.linalg import sqrtm
from functools import partial

@partial(jit, static_argnums=(0,1,2))
def generate_chain(k, length, d, μ, Σ_sqrt, transition_probs, initial_probs, key):
    key, keyc, keyn = random.split(key, 3)
    initial_state = random.choice(keyc, k, p=initial_probs)

    initial_observation = random.normal(keyn, d) @ Σ_sqrt[initial_state] + μ[initial_state]

    def step(key_state, _):
        key, state = key_state
        key, keyc, keyn = random.split(key,3)
        state = random.choice(keyc, k, p=dynamic_index_in_dim(transition_probs,state).squeeze())
        observation = random.normal(keyn, d) @ dynamic_index_in_dim(Σ_sqrt, state) + dynamic_index_in_dim(μ, state)
        return (key, state), (state, observation)
    _, (real_chain, observations) = scan(step, (key, initial_state), np.arange(length-1))

    real_chain = jnp.concatenate((jnp.array([initial_state]), real_chain))
    observations = jnp.concatenate((jnp.array([[initial_observation]]), observations))
    print(observations.shape, real_chain.shape)
    return (real_chain, observations.squeeze()) #, np.array([key for key, _ in groupby(real_chain)]))

def generate_chains(states : int, amount, length = 30, d=3,
        μ : None | np.ndarray=None,
        Σ : None | np.ndarray=None,
        transition_probs : None | np.ndarray = None,
        initial_probs : None | np.ndarray = None):
    k = states
    if transition_probs is None:
        transition_probs = np.random.uniform(0, 1, size=(k,)*2) + np.eye(k)*2
        transition_probs = (transition_probs / np.sum(transition_probs, axis=0)).T
    else:
        assert(transition_probs.shape == (k,k))
    if initial_probs is None:
        initial_probs = np.random.uniform(0, 1, size=(k))
        initial_probs /= np.sum(initial_probs)
    else:
        assert(len(initial_probs) == k)
    if μ is None:
        μ = np.random.normal(size=(k,d))
    if Σ is None:
        Σ_sqrt = np.random.normal(size=(k,d,d))
        Σ = Σ_sqrt.transpose((0,2,1)) @ Σ_sqrt
    Σ_sqrt = np.array([sqrtm(Σi) for Σi in Σ])
    key = random.key(0)
    keys = random.split(key, amount)

    chains = [generate_chain(k, length, d, μ, Σ_sqrt, transition_probs, initial_probs, key) for key in keys]

    def unduplicate(chain):
        mask = np.ones(len(chain), dtype=bool)
        mask[1:] = chain[1:] != chain[:-1]
        return chain[mask]
    chains = [chain + (unduplicate(np.array(chain[0])),) for chain in chains]
    return chains
