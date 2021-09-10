"""
Vector Autoregressive Moving Average with eXogenous regressors model = VARMAX(p, q)
"""
# In[] Libs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.preprocessing import MaxAbsScaler
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# In[] Datasets
_dataset_dir = "GaitData"
_cut_from = 1200
_cut_to = 2300
_dataset_file = "02_RightAnkle.csv"
df_RA = pd.read_csv(f"..\\datasets\\{_dataset_dir}\\{_dataset_file}", skiprows=12)[_cut_from:_cut_to].reset_index()
df_RA.drop(columns=["PacketCounter", "StatusWord", "ClippingFlags", "RSSI"], inplace=True)
_dataset_file = "02_LeftAnkle.csv"
df_LA = pd.read_csv(f"..\\datasets\\{_dataset_dir}\\{_dataset_file}", skiprows=12)[_cut_from:_cut_to].reset_index() * -1 # Multiply by -1 to correct the signal from left ankle
df_LA.drop(columns=["PacketCounter", "StatusWord", "ClippingFlags", "RSSI"], inplace=True)

# In[] Plot the dataset
_cols_to_plot = ["Gyr_Z"]
plt.clf()
plt.plot(df_RA[_cols_to_plot], label="Gyr_Z_RA")
plt.plot(df_LA[_cols_to_plot], label="Gyr_Z_LA")
plt.legend()
plt.show()

# In[] Add new params
df_RA["FreeAcc_mag"] = (df_RA["FreeAcc_E"]**2 + df_RA["FreeAcc_N"]**2 + df_RA["FreeAcc_U"]**2)**.5
df_RA["Gyr_mag"] = (df_RA["Gyr_X"]**2 + df_RA["Gyr_Y"]**2 + df_RA["Gyr_Z"]**2)**.5

df_LA["FreeAcc_mag"] = (df_LA["FreeAcc_E"]**2 + df_LA["FreeAcc_N"]**2 + df_LA["FreeAcc_U"]**2)**.5
df_LA["Gyr_mag"] = (df_LA["Gyr_X"]**2 + df_LA["Gyr_Y"]**2 + df_LA["Gyr_Z"]**2)**.5

# In[] Train and Test the model by splitting the data set into Train and Test sets
_col_to_workon = "Gyr_Z"
df_joined = pd.concat((df_RA[_col_to_workon], df_LA[_col_to_workon]), axis=1)
df_joined.columns = [_col_to_workon+"_RA", _col_to_workon+"_LA"]
df_joined["timeframe(ms)"] = np.linspace(start=0, stop=len(df_joined)-1, num=len(df_joined), dtype=np.int16)
df_joined.set_index(df_joined["timeframe(ms)"], inplace=True)
df_joined.drop(["timeframe(ms)"], axis=1, inplace=True)
df_joined.index.freq = "ms"
_train_test_ratio = 0.7 # equals to 70%
_n_cutoff = int(len(df_joined) * _train_test_ratio)
_n_test = len(df_joined) - _n_cutoff

train_set = df_joined.iloc[:-_n_test].copy()
test_set = df_joined.iloc[-_n_test:].copy()

# In[] Scale the data
scaler = MaxAbsScaler()
train_set[[_col_to_workon+"_RA_scaled", _col_to_workon+"_LA_scaled"]] = scaler.fit_transform(X=train_set)
test_set[[_col_to_workon+"_RA_scaled", _col_to_workon+"_LA_scaled"]] = scaler.fit_transform(X=test_set)

# In[] Indexing
train_index = df_joined.index <= train_set.index[-1]
test_index = df_joined.index > train_set.index[-1]

df_joined.loc[train_index, _col_to_workon+"_RA_scaled"] = train_set[_col_to_workon+"_RA_scaled"]
df_joined.loc[train_index, _col_to_workon+"_LA_scaled"] = train_set[_col_to_workon+"_LA_scaled"]

df_joined.loc[test_index, _col_to_workon+"_RA_scaled"] = test_set[_col_to_workon+"_RA_scaled"]
df_joined.loc[test_index, _col_to_workon+"_LA_scaled"] = test_set[_col_to_workon+"_LA_scaled"]

# In[] Fit the model
_cols_to_workon = [_col_to_workon+"_RA_scaled", _col_to_workon+"_LA_scaled"]

# Plot PACF ACF to determine the value of p and q
fig, axes = plt.subplots(2, 1)
plot_pacf(x=train_set[_cols_to_workon[0]], ax=axes[0]) # to determine the value of p
plot_acf(x=train_set[_cols_to_workon[0]], ax=axes[1]) # to determine the value of q
plt.show()

_time_start = datetime.now()
model = VARMAX(endog=train_set[_cols_to_workon], order=(1, 1)).fit(maxiter=100)
print(f"Duration: {datetime.now() - _time_start}")

# In[] Plot the results
df_joined.loc[train_index, _col_to_workon+"_RA_Train_Pred"] = model.fittedvalues[_col_to_workon+"_RA_scaled"]
df_joined.loc[test_index, _col_to_workon+"_RA_Test_Pred"] = model.get_forecast(_n_test).predicted_mean[_col_to_workon+"_RA_scaled"]

_cols_to_plot = [_col_to_workon+"_RA_scaled", _col_to_workon+"_RA_Train_Pred", _col_to_workon+"_RA_Test_Pred"]
df_joined[_cols_to_plot].plot()
