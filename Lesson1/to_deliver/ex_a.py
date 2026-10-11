from jax import vmap, jnp, random as rnd

seed = 1701
k = rnd.key(seed)
inp = rnd.normal(k, (10, 1))

def compute_arctan(x):
    return jnp.arctan(x)

compute_arctan_v = vmap(compute_arctan, in_axes=(0)) # il numero qui indica l'asse sul quale si trova il nostro vettore di input

result = compute_arctan_v(inp)

print(result)

