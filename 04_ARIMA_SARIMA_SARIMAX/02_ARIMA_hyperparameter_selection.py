"""
Stationarity Test
    Set d based on the stationarity
Partial Autocorrelation Function (PACF) - to select p for AR(p)
    Set p = maximum non-zero (outside confidence interval) lag 
Autocorrelation Function (ACF) - to find q for MA(q)
    Set q = maximum non-zero (outside confidence interval) lag
    
Resource:
    https://www.youtube.com/watch?v=5Q5p6eVM7zM&ab_channel=DataScienceShow
"""
# In[] Libs
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# In[] define custom methods
def stationarity_test(data, alpha_level=0.05):
    # Note: uncomment the codes to debug the function.
    from statsmodels.tsa.stattools import adfuller
    
    res = adfuller(data.dropna().copy())
    # print(f"Test-Statistic: {res[0]:0.2f}")
    # print(f"P-Value: {res[1]:0.2f}")
    if res[1] < alpha_level:
        print("Stationary")
        return True
    else:
        print("Non-Stationary")
        return False
        
def set_d_value(data, alpha_level=0.05):
    stationarity = False
    d = 0
    while (not stationarity):
        print(f"d is: {d}")
        stationarity = stationarity_test(data=data, alpha_level=alpha_level)
        if stationarity:
            print(f"d: {d}")
            return d
        else:
            data = data.diff()
            d += 1
    print(f"d final is: {d}")
    return d
        
# In[] Datasets
_dataset_dir = "GaitData"
_dataset_file = "02_RightAnkle.csv"
df_stationary = pd.read_csv(f"..\\datasets\\{_dataset_dir}\\{_dataset_file}", skiprows=12)[1200:2300].reset_index()

_dataset_dir = "StockData"
_dataset_file = "airline_passengers.csv"
df_non_stationary = pd.read_csv(f"..\\datasets\\{_dataset_dir}\\{_dataset_file}", skiprows=0)

# In[] Plot the datasets
fig, axes = plt.subplots(2, 1)
_cols_to_plot = ["Gyr_Z"]
axes[0].plot(df_stationary[_cols_to_plot])
axes[0].legend(_cols_to_plot)
_cols_to_plot = ["Passengers"]
axes[1].plot(df_non_stationary[_cols_to_plot])
axes[1].legend(_cols_to_plot)
plt.show()

# In[] Plot the PACF and ACF functions to determine p and q hyperparameters
_col_to_workon = "Gyr_Z"
fig, axes = plt.subplots(2, 1)
plot_pacf(x=df_stationary[_col_to_workon], ax=axes[0])
plot_acf(x=df_stationary[_col_to_workon], ax=axes[1])
plt.show()

set_d_value(data=df_stationary[_col_to_workon], alpha_level=0.05)

# In[] Plot the PACF and ACF functions to determine p and q hyperparameters
_col_to_workon = "Passengers"
fig, axes = plt.subplots(2, 1)
plot_pacf(x=df_non_stationary[_col_to_workon], ax=axes[0])
plot_acf(x=df_non_stationary[_col_to_workon], ax=axes[1])
plt.show()

set_d_value(data=df_non_stationary[_col_to_workon], alpha_level=0.05)

# In[] Finish