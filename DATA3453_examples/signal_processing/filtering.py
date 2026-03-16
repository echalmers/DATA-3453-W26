import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fft import fft


# sample data with yearly cyclical trend
df = pd.DataFrame()
df['date'] = pd.date_range(start="2024-01-01",
                      periods=104,
                      freq="W")
df['sales'] = pd.Series(np.sin(2 * np.pi * 1 / 52 * df.index)) * 100 + 150

# add noise
df['sales'] += pd.Series(np.sin(2 * np.pi * 20 / 52 * df.index)) * 50 # a high-frequency signal
# df['sales'] += np.random.normal(0, 50, len(df.index)) # random noise


# plot sample data
plt.subplot(1, 4, 1)
plt.plot(df['date'], df['sales'], 'k')
plt.ylabel('sales ($k)')
plt.title('sales over two years')
plt.tick_params(axis='x', labelrotation=45)

# look at frequency content of signal
plt.subplot(1, 4, 2)
frequency_content = np.abs(fft(df['sales']))
plt.plot(np.linspace(0, 1, 52), frequency_content[:52])
plt.title('frequency content of sales data')
plt.xlabel('normalized frequency')

# create coefficients for 3 rolling averages and a low-pass filter
rolling_avg_coefs_9 = [1/9] * 9
rolling_avg_coefs_33 = [1/33] * 33
low_pass_filter_9 = signal.firwin(9, 
                      cutoff=0.1, # normalized freqency where response starts cutting off
                      pass_zero='lowpass' # 'lowpass' or 'highpass'
                      )

# plot responses of the filters
plt.subplot(1, 4, 3)
normalized_freq, response_avg_9 = signal.freqz(rolling_avg_coefs_9, fs=2)
plt.plot(normalized_freq, np.abs(response_avg_9))

normalized_freq, response_avg_33 = signal.freqz(rolling_avg_coefs_33, fs=2)
plt.plot(normalized_freq, np.abs(response_avg_33))

normalized_freq, response_filter_9 = signal.freqz(low_pass_filter_9, fs=2)
plt.plot(normalized_freq, np.abs(response_filter_9))
plt.title('frequency reponses of various filters')
plt.legend(['rolling average (k=9)', 'rolling average (k=33)', 'lowpass filter (k=9)'])
plt.xlabel('normalized frequency')
plt.ylabel('magnitude response')


# apply filters and plot the results
plt.subplot(1, 4, 4)
roll_averaged = df['sales'].rolling(window=9, center=True).agg(lambda window: np.dot(window, rolling_avg_coefs_9))
filtered = df['sales'].rolling(window=9, center=True).agg(lambda window: np.dot(window, low_pass_filter_9))

plt.plot(df['date'], df['sales'], 'k', alpha=0.25)
plt.plot(df['date'], roll_averaged)
plt.plot(df['date'], filtered, 'g')
plt.ylabel('sales ($k)')
plt.title('sales data before and after filtering')
plt.tick_params(axis='x', labelrotation=45)
plt.legend(['sales data', 'rolling average (k=9)', 'lowpass filter (k=9)'])
plt.show()
exit()


def my_custom_func(x):
    return 1


my_series.rolling(window=9, center=True).mean() # rolling average, size-9 window
my_series.rolling(window=9, center=True).agg(my_custom_func) # a custom rolling aggregation


# perform filtering
window_size = 9
from scipy import signal
coefs = signal.firwin(window_size, 
                      cutoff=0.5, # normalized freqency where response starts cutting off
                      pass_zero='lowpass' # 'low-pass' or 'high-pass'
                      )
# coefs = np.array([-1, 0, 1, 2, 4, -4, 5, 6, 10]).astype(float)  #np.random.rand(9)
# coefs /= coefs.sum()
data_avg = df['sales'].rolling(window_size, center=True).mean()
data_filt = df['sales'].rolling(window_size, center=True).agg(lambda x: np.dot(coefs, x))

from scipy import signal
normalized_freq, response_avg = signal.freqz([1/window_size] * window_size, fs=2)
normalized_freq, response = signal.freqz(coefs, fs=2)
plt.plot(normalized_freq, np.abs(f_avg))
# plt.plot(np.linspace(0, 1, len(f_filt)), np.abs(f_filt))
# plt.legend(['avg', 'filt'])
plt.xlabel('normalized frequency')
plt.ylabel('magnitude after averaging')
plt.title('frequency response of a rolling average')
plt.show()

# plot
plt.plot(df['date'], df['sales'])
plt.plot(df['date'], data_avg)
plt.plot(df['date'], data_filt)
plt.legend(['sales', 'avg', 'filt'])
plt.show()
