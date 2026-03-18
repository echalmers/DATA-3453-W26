from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import numpy as np


# generate some 2d data, with a strong correlation between the dimensions
cov = [[1, 0.9],
       [0.9, 1]]
data = np.random.multivariate_normal([0, 0], cov, size=50)

# use PCA and T-SNE to reduce to 1 dimension
pca = PCA(n_components=1)
data_pca = pca.fit_transform(data)
tsne = TSNE(n_components=1)
data_tsne = tsne.fit_transform(data)

# plot the results
fig, ax = plt.subplots(1, 3)
ax[0].scatter(data[:, 0], data[:, 1], marker='x', c=data.sum(axis=1))
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title('original data')

ax[1].scatter(data_pca, np.zeros_like(data_pca), marker='x', c=data.sum(axis=1))
ax[1].yaxis.set_visible(False)
ax[1].set_xlabel('x\'')
ax[1].set_title('PCA principle component')

ax[2].scatter(data_tsne, np.zeros_like(data_tsne), marker='x', c=data.sum(axis=1))
ax[2].yaxis.set_visible(False)
ax[2].set_xlabel('new x')
ax[2].set_title('T-SNE principle component')

plt.show()


# now load the MNIST digits dataset
digits = load_digits(n_class=6)
X, y = digits.data, digits.target

fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))
for idx, ax in enumerate(axs.ravel()):
    ax.imshow(X[idx].reshape((8, 8)), cmap=plt.cm.binary)
    ax.axis("off")
_ = fig.suptitle("A selection from the 64-dimensional digits dataset", fontsize=16)
plt.show()


# Use PCA and T-SNE to reduce the high-dimensional digits to 2-dimensions so they can be plotted
# YOUR CODE HERE