from statistics import covariance

from jax import numpy as jnp, random as rnd

seed = 1701
k1,k2,key = rnd.split(rnd.key(seed),3) # unless they will have the same value
num_features = 40
num_gauss = 42

# A = jnp.tile((rnd.choice(k2, num_features, shape=(num_features,), replace=False)/jnp.sum(rnd.choice(k2, num_features, shape=(num_features,), replace=False)))[None,:],(num_features,1))
# # print(A.shape[0],A.shape[1])

# pi = rnd.choice(k2, num_features, shape=(num_features,), replace=False)/jnp.sum(rnd.choice(k2, num_features, shape=(num_features,), replace=False))
# print(pi)
# print(jnp.sum(pi))

# responsability = jnp.zeros((num_features,num_gauss))
# print(responsability.shape[0],responsability.shape[1])


# idx = rnd.choice(k1, num_gauss, shape=(num_gauss,), replace=False)
# print(sentence[idx])
#
# gauss = {
#     "covariance": jnp.tile(jnp.cov(num_features, rowvar=False)[None,:,:],(num_gauss,1,1)),
# }

# print(gauss["covariance"])
print(jnp.sum(jnp.ones(num_gauss) / num_gauss,))
