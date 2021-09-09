"""
AutoRegressive Integrated Moving Average (ARIMA)
ARIMA(p, d, q)
    p: referes to the autoregressive part - AR(p)
    q: referes to the moving average part - MA(q)
    d: referes to the integrated part - I(d)
    Note:
        ARIMA(p, 0, 0) == AR(p) OR ARMA(p, 0)
        ARIMA(0, 0, q) == ARMA(0, q) AND MA(q)
        ARIMA(0, d, 0) == I(d)
        ARIMA(0, 1, 0) == I(1) == Random Walk
"""
# In[] Libs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from utils.custom_functions import plot_fit_and_forecast

# In[] Datasets
_dataset_dir = "GaitData"
_dataset_file = "02_RightAnkle.csv"
df = pd.read_csv(f"..\\datasets\\{_dataset_dir}\\{_dataset_file}", skiprows=12)
df.drop(columns=["PacketCounter", "StatusWord", "ClippingFlags", "RSSI"], inplace=True)

# In[] Plot the dataset
_cols_to_plot = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
plt.clf()
plt.plot(df[_cols_to_plot])
plt.legend(_cols_to_plot)
plt.show()

# In[] Add new params
df["FreeAcc_mag"] = (df["FreeAcc_E"]**2 + df["FreeAcc_N"]**2 + df["FreeAcc_U"]**2)**.5
df["Gyr_mag"] = (df["Gyr_X"]**2 + df["Gyr_Y"]**2 + df["Gyr_Z"]**2)**.5

# In[] Add other params
_col_to_workon = "Gyr_Z"
df[_col_to_workon+"_1stDiff"] = df[_col_to_workon].diff() # Calculate the different to check the seasonality and the variation
plt.clf()
plt.plot(df[_col_to_workon], label=_col_to_workon)
plt.plot(df[_col_to_workon+"_1stDiff"], label=_col_to_workon+"_1stDiff")
plt.legend()
plt.show()

# In[] Data Analysis
df[_col_to_workon+"_log"] = np.log(np.abs(df[_col_to_workon]) + 1) # convert the data to positive non-zero signal
plt.clf()
plt.plot(df[_col_to_workon], label=_col_to_workon)
plt.plot(df[_col_to_workon+"_1stDiff"], label=_col_to_workon+"_1stDiff")
plt.plot(df[_col_to_workon+"_log"], label=_col_to_workon+"_log")
plt.legend()
plt.show()

# In[] Train and Test the model by splitting the data set into Train and Test sets
_train_test_ratio = 0.8 # equals to 80%
_n_cutoff = int(len(df) * _train_test_ratio)
_n_test = len(df) - _n_cutoff
df_train = df[_col_to_workon].iloc[:_n_cutoff]
df_test = df[_col_to_workon].iloc[_n_cutoff:]

arima_1_0_0 = ARIMA(df[_col_to_workon], order=(1, 0, 0)).fit()

df.loc[:_n_cutoff-1, _col_to_workon+"arima_1,0,0"] = arima_1_0_0.fittedvalues[:_n_cutoff].values # which is equla to arima_1.predict(start=df_train.index[0], end=df_train.index[-1])
df.loc[_n_cutoff:, _col_to_workon+"arima_1,0,0"] = arima_1_0_0.forecast(_n_test).values # or df.loc[_n_cutoff:, _col_to_workon+"arima_1"] = arima_1.get_forecast(_n_test).predicted_mean.values

# In[] Plot the predictions
_cols_to_plot = [_col_to_workon, _col_to_workon+"arima_1,0,0"]
plt.clf()
plt.plot(df[_cols_to_plot])
plt.legend(_cols_to_plot)
plt.show()

# In[]
arima_forcast_model = arima_1_0_0.get_forecast(_n_test)
print(arima_forcast_model.conf_int()) # check the confidence intervals

# In[] Plot the predictions with confidence intervals
plot_fit_and_forecast(df_main=df[_col_to_workon], df_train=df_train, df_test=df_test, forecast_model=arima_1_0_0)

# In[] Test other hyperparams fpr ARIMA -> ARIMA(10, 0, 0)
arima_10_0_0 = ARIMA(df[_col_to_workon], order=(10, 0, 0)).fit()
plot_fit_and_forecast(df_main=df[_col_to_workon], df_train=df_train, df_test=df_test, forecast_model=arima_10_0_0)

# In[] Test other hyperparams fpr ARIMA -> ARIMA(0, 0, 10)
arima_0_0_10 = ARIMA(df[_col_to_workon], order=(0, 0, 10)).fit()
plot_fit_and_forecast(df_main=df[_col_to_workon], df_train=df_train, df_test=df_test, forecast_model=arima_0_0_10)

# In[] Test other hyperparams fpr ARIMA -> ARIMA(8, 1, 1)
arima_8_1_1 = ARIMA(df[_col_to_workon], order=(8, 1, 1)).fit()
plot_fit_and_forecast(df_main=df[_col_to_workon], df_train=df_train, df_test=df_test, forecast_model=arima_8_1_1)

# In[] Test other hyperparams fpr ARIMA -> ARIMA(8, 1, 1)
arima_8_1_1 = ARIMA(df[_col_to_workon+"_log"], order=(20, 1, 1)).fit() # test on logged data
plot_fit_and_forecast(df_main=df[_col_to_workon+"_log"], df_train=df_train, df_test=df_test, forecast_model=arima_8_1_1)

