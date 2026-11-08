from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import meteostat as ms
np.set_printoptions(threshold=np.inf)

POINT = ms.Point(1.1494, 104.0249)
START = date(2025, 1, 1)
END = date(2025, 12, 31)
stations = '48698'
'''(yang atas)Generate 365 daily temperature
 readings using a 
 (yang bawah) normal distribution (mean 28, std 4) plus random noise.'''
random_distribution = np.random.normal(loc=28, scale=4, size=None)

'''Smooth seasonal noise using a 7-day moving average with convolution'''
# ms.daily() creates a TimeSeries query object configured with stations and date range
# .fetch() executes HTTP request and returns a pandas DataFrame
# ['temp'] extracts the temperature column as a pandas Series
# .to_numpy() converts the Series into a 1D NumPy ndarray
temps = ms.daily(stations, START, END).fetch()['temp'].dropna() # fetch untuk download untuk ngetrigger
''' np.convolve(a, v, mode='valid') computes rolling linear combination
temps: 1D array of daily temperature input values
np.ones(7)/7: kernel weights of 7 equal values (1/7 each) to calculate an unweighted 7-day mean
mode='valid': calculates only where window completely overlaps, dropping 6 edge points without padding'''
sma7 = np.convolve(temps, np.ones(7)/7, mode='valid')

'''Slice original temperatures to align shape with smoothed output'''
aligned_temps = temps[6:]

'''Detect heatwave days: find indices where temperature exceeds the 
smoothed average by at least 4 degrees using boolean masking or nonzero.'''
heatwave = np.nonzero(aligned_temps >= (sma7 + 3))

trim = temps[1:].to_numpy().reshape(52, 7)

plt.plot(aligned_temps.to_numpy(), label='Actual Temp', alpha=0.6)
plt.plot(sma7, label='7-Day SMA', color='orange')
plt.plot(sma7 + 3, label='Threshold (+3)', color='red', linestyle='--')
plt.scatter(heatwave[0], aligned_temps.iloc[heatwave[0]], color='red', label='Heatwave')
plt.legend()
plt.show()