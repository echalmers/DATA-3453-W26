import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def bootstrap_CI(values, n_resamples, CI_lower=2.5, CI_upper=97.5, statistic_to_calculate=np.mean):
    statistics = []
    for resample in range(n_resamples):
        resample = np.random.choice(values, len(values), replace=True) # sample values with replacement
        statistics.append(statistic_to_calculate(resample))
    return np.percentile(statistics, CI_lower), np.percentile(statistics, CI_upper)


if __name__ == '__main__':
    df = pd.read_csv('./DATA3453_examples/plotly_dash/housing_holdout.csv', index_col=None)

    mean_price = np.mean(df['re_assessed_value_2024'])
    lower, upper = bootstrap_CI(df['re_assessed_value_2024'], 1000, 2.5, 97.5)

    print(mean_price, lower, upper)

    plt.bar(1, mean_price)
    plt.errorbar(x=1, y=mean_price, yerr=[[mean_price - lower], [upper - mean_price]], capsize=10, color='k')
    plt.show()