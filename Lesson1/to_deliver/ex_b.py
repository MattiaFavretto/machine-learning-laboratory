from jax import grad, vmap, numpy as jnp, random as rnd
import matplotlib.pyplot as plt

# Computing the gradient using grad function
seed = 1701
k = rnd.key(seed)
inp = rnd.normal(k, (10,))

def compute_arctan(x):
    return jnp.arctan(x)

compute_actan_grad = grad(compute_arctan, argnums=(0), allow_int=True)
compute_arctan_vmap = vmap(compute_actan_grad, in_axes=(0))

result_grad = compute_arctan_vmap(inp)
print(result_grad)

# Computing the gradient using delta
result_delta = []
for i in range(len(inp)):
    result_delta.append(1/(1+inp[i]**2))

print(result_delta)

# PLOTTING RESULTS
fig, ax = plt.subplots(figsize=(10,10))

# different dimensions for the point in order to see them even if they are in the same place
ax.scatter(inp,result_grad,color="red", s=50, label="Grad function")
ax.scatter(inp,result_delta,color="blue",s=10, label="Delta function")

ax.set_xlabel("X")
ax.set_ylabel("Gradient")
ax.grid(True)

plt.show()
