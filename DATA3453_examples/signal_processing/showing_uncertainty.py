import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from bootstrapping import bootstrap_CI
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
df = df[(df['x'] < 7) | (df['x'] > 13)]
df.index = np.arange(len(df))
df.loc[len(df), ['x', 'y']] = [10, 10**0.8 + 0.2]
df.loc[len(df), ['x', 'y']] = [10, 10**0.8 - 0.2]
df.loc[len(df), ['x', 'y']] = [20, 20 ** 0.8 - 4]

# plot the data
plt.scatter(df.x, df.y, c=['b']*(len(df)-3) + ['r'] * 3, alpha=[0.1]*(len(df)-3) + [0.5]*3)
sns.lineplot(df, x='x', y='y', errorbar=('ci', 95))
plt.title('original data with Seaborn 95% CI')


# fit a KDE to the data.
kde = KernelDensity(bandwidth=2)
kde.fit(df[['x']])

# plot the KDE estimates of P(x)
fig, ax = plt.subplots(2, 1, sharex=True)
x_vals = np.linspace(0, 40, 100)
ax[1].plot(x_vals, np.exp(kde.score_samples(x_vals.reshape(-1, 1))))
ax[1].set_title('KDE estimates of P(x)')

# re-create the CI (using bootstrapping), but now divide the CI by KDE estimates of P(x)
x_vals = df['x'].sort_values().unique()
y_mean_vals = df.groupby('x')['y'].mean().values
kde_probability = np.exp(kde.score_samples(x_vals.reshape(-1, 1)))
kde_probability_normalized = kde_probability / kde_probability.max()

CIs = np.array([bootstrap_CI(df.loc[df['x'] == x, 'y'], n_resamples=100) for x in x_vals])
# push the upper and lower CI limits away from the mean line
CIs -= y_mean_vals.reshape(-1, 1)
CIs[:, 0] /= (kde_probability_normalized ** 2)
CIs[:, 1] /= (kde_probability_normalized ** 2)
CIs += y_mean_vals.reshape(-1, 1)

# plot the new CIs
plt.sca(ax[0])
plt.scatter(df['x'], df['y'], c='b', alpha=0.2)
plt.plot(x_vals, y_mean_vals)
plt.fill_between(x_vals, CIs[:, 0], CIs[:, 1], alpha=0.5)


# fit a gaussian process to the data
kernel = RationalQuadratic(length_scale=1) 
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True) 
gp.fit(df[['x']], df[['y']])
y_mean, y_std = gp.predict(np.linspace(0, 40, 100).reshape(-1, 1), return_std=True)


# plot the mean and standard deviation predicted by the guassian process regression
plt.figure()
plt.scatter(df.x, df.y, alpha=0.2)
plt.plot(np.linspace(0, 40, 100), y_mean)
plt.fill_between(np.linspace(0, 40, 100), y_mean - y_std, y_mean + y_std, alpha=0.5)

plt.show()