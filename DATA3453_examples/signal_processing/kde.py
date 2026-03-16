import pandas as pd
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt


# read in the table
df = pd.read_csv('grades_data.csv', index_col=None)
df['avg_assignment'] = df[
   ['Assignment 1','Assignment 2','Assignment 3','Assignment 4']
].mean(axis=1)

# build the KDE
kde = KernelDensity(bandwidth=0.75) # tune the bandwidth as needed
kde.fit(df[['avg_assignment', 'Final Exam']])

# construct a matrix showing probabilities for every combination of scores
# i.e. matrix[i, j] shows probability of scoring i on assignments, and j on the final
probabilities = # YOUR CODE HERE

# plot
plt.imshow(probabilities, extent=[0, 1, 0, 1], origin='lower', cmap='Blues', interpolation='bilinear', alpha=0.5)
plt.scatter(df['Final Exam'], df['avg_assignment'], c='blue')
plt.show()