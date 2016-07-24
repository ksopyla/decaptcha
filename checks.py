import numpy as np
import matplotlib.pyplot as plt
import vec_mappings as vecmp


batch_idx=0

batch_xs, batch_ys, idx = vecmp.random_batch(X, Y, batch_size)


k=idx[batch_idx]

np.sum(X[k,:]!= batch_xs[batch_idx,:])
np.sum(Y[k,:]!= batch_ys[batch_idx,:])

captcha_text[k]
vecmp.map_vec2words(batch_ys[batch_idx,:])


plt.imshow(X[k,:].reshape(64,304),cmap='Greys_r')
plt.imshow(batch_xs[batch_idx,:].reshape(64,304),cmap='Greys_r')