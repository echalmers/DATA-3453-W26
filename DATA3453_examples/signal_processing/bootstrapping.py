import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def bootstrap_CI(values, n_resamples, CI_lower, CI_upper):
    statistics = []
    for resample in range(n_resamples):
        resample = np.random.choice(values, len(values), replace=True) # sample values with replacement
        statistics.append(np.percentile(resample, 75))
    return np.percentile(statistics, CI_lower), np.percentile(statistics, CI_upper)

df = pd.read_csv('./DATA3453_examples/plotly_dash/housing_holdout.csv', index_col=None)

p75_price = np.percentile(df['re_assessed_value_2024'], 75)
lower, upper = bootstrap_CI(df['re_assessed_value_2024'], 1000, 2.5, 97.5)

print(p75_price, lower, upper)

plt.bar(1, p75_price)
plt.errorbar(x=1, y=p75_price, yerr=[[p75_price - lower], [upper - p75_price]], capsize=10, color='k')
plt.show()