import pandas as pd
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
import numpy as np


# read in the table
df = pd.read_csv('./DATA3453_examples/signal_processing/grades_data.csv', index_col=None)
df['avg_assignment'] = df[
   ['Assignment 1','Assignment 2','Assignment 3','Assignment 4']
].mean(axis=1)

# build the KDE
kde = KernelDensity(bandwidth=0.15) # tune the bandwidth as needed
kde.fit(df[['avg_assignment', 'Final Exam']])

# construct a matrix showing probabilities for every combination of scores
# i.e. matrix[i, j] shows probability of scoring i on assignments, and j on the final
x_vals = np.linspace(0, 1, 50)
y_vals = np.linspace(0, 1, 50)
probabilities = np.zeros((50, 50))
for i in range(50):
   for j in range(50):
      probabilities[i, j] = np.exp(kde.score_samples([[x_vals[i], y_vals[j]]]))[0]

# plot
plt.imshow(probabilities, extent=[0, 1, 0, 1], origin='lower', cmap='Blues', interpolation='bilinear', alpha=0.5)
plt.scatter(df['Final Exam'], df['avg_assignment'], c='blue')
plt.show()