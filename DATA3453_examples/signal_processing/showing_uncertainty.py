import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from bootstrapping import boostrap_CI
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic


# generate some sample, noisy data
n = 400
df = pd.DataFrame()
df['x'] = np.random.randint(0, 40, n)
df['y'] = df['x'] ** 0.8 + np.random.normal(0, 3, n)

# create a few places with sparse data
df = df[(df['x'] != 10) & (df['x'] != 20)]
df = df[(df['x'] < 25) | (df['x'] > 30)]
df.index = np.arange(len(df))
df.loc[len(df), ['x', 'y']] = [10, 10**0.8 + 0.2]
df.loc[len(df), ['x', 'y']] = [10, 10**0.8 - 0.2]
df.loc[len(df), ['x', 'y']] = [20, 20 ** 0.8 - 4]

# plot the data
fig, ax = plt.subplots(2, 1, sharex=True)
plt.sca(ax[0])
plt.scatter(df.x, df.y, c=['b']*(len(df)-3) + ['r'] * 3, alpha=[0.1]*(len(df)-3) + [0.5]*3)
sns.lineplot(df, x='x', y='y', errorbar=('ci', 95))
plt.title('original data with Seaborn 95% CI')
