import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./DATA3453_examples/signal_processing/noisy series data.csv', index_col=None)
plt.scatter(df['X'], df['Y'], alpha=0.2)
sns.lineplot(df, x='X', y='Y', errorbar=('ci', 95), estimator='median')
plt.show()