"""
Holt-Winter - Exponential Smoothing
"""
# In[] Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# In[] Datasets
_dataset_dir = "GaitData"
_dataset_file = "02_LeftAnkle.csv"
df = pd.read_csv(f"..\\datasets\\{_dataset_dir}\\{_dataset_file}", skiprows=12)
df.drop(columns=["PacketCounter", "StatusWord", "ClippingFlags", "RSSI"], inplace=True)

# In[] Plot the dataset
_cols_to_plot = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
plt.clf()
plt.plot(df[_cols_to_plot])
plt.legend(_cols_to_plot)
plt.show()

# In[] Add new Params
df["FreeAcc_mag"] = (df["FreeAcc_E"]**2 + df["FreeAcc_N"]**2 + df["FreeAcc_U"]**2)**.5
df["Gyr_mag"] = (df["Gyr_X"]**2 + df["Gyr_Y"]**2 + df["Gyr_Z"]**2)**.5

# In[] Data Analysis - Smoothing and Fitting
_col_to_workon = "Gyr_Z"
holt_ES = ExponentialSmoothing(
    endog=df[_col_to_workon],
    initialization_method="legacy-heuristic",
    trend="add",
    seasonal="add",
    seasonal_periods=100
).fit()

df[_col_to_workon+"_ES"] = holt_ES.predict(start=df[_col_to_workon].index[0], end=df[_col_to_workon].index[-1])

if np.allclose(df[_col_to_workon+"_ES"], holt_ES.fittedvalues):
    print("Predicted values are close.")
else:
    print("Predicted values are NOT close.")

# In[] Plot the results all together
_cols_to_plot = [_col_to_workon, _col_to_workon+"_ES"]
plt.clf()
plt.plot(df[_cols_to_plot])
plt.legend(_cols_to_plot)
plt.show()

# In[] Train and Test the model by splitting the data set into Train and Test sets
_train_test_ratio = 0.8 # equals to 80%
_n_cutoff = int(len(df) * _train_test_ratio)
_n_test = len(df) - _n_cutoff
df_train = df[_col_to_workon].iloc[:_n_cutoff]
df_test = df[_col_to_workon].iloc[_n_cutoff:]

holt_ES_2 = ExponentialSmoothing(
    endog=df[_col_to_workon],
    initialization_method="legacy-heuristic",
    trend="add",
    seasonal="add",
    seasonal_periods=100
).fit()

df.loc[:_n_cutoff-1, _col_to_workon+"ES_fitted"] = holt_ES_2.fittedvalues[:_n_cutoff].values
df.loc[_n_cutoff:, _col_to_workon+"ES_fitted"] = holt_ES_2.forecast(_n_test).values

# In[] Plot the predictions
_cols_to_plot = [_col_to_workon, _col_to_workon+"ES_fitted"]
plt.clf()
plt.plot(df[_cols_to_plot])
plt.legend(_cols_to_plot)
plt.show()

# Print the results
for k, val in zip(holt_ES_2.params.keys(), holt_ES_2.params.values()):
    print(f"{k}: {val}")

