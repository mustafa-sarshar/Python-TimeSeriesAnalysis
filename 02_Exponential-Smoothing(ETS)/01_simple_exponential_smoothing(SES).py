"""
Simple Moving Average (SMA)
"""
# In[] Libs
import pandas as pd
import matplotlib.pyplot as plt

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

# In[] Data Analysis
_col_to_workon = "Gyr_Z"
_windows_size = 10
df_rolling = df.rolling(window=_windows_size)
df[_col_to_workon+"_rolling_mean"] = df_rolling[_col_to_workon].mean()
df[_col_to_workon+"_rolling_median"] = df_rolling[_col_to_workon].median()
df[_col_to_workon+"_rolling_var"] = df_rolling[_col_to_workon].var()
df[_col_to_workon+"_rolling_std"] = df_rolling[_col_to_workon].std()

# In[] Plot the results
fig, axes = plt.subplots(5, 1)
axes[0].plot(df[_col_to_workon])
axes[0].set_title(_col_to_workon)
axes[1].plot(df[_col_to_workon+"_rolling_mean"])
axes[1].set_title(_col_to_workon+"_rolling_mean")
axes[2].plot(df[_col_to_workon+"_rolling_median"])
axes[2].set_title(_col_to_workon+"_rolling_median")
axes[3].plot(df[_col_to_workon+"_rolling_var"])
axes[3].set_title(_col_to_workon+"_rolling_var")
axes[4].plot(df[_col_to_workon+"_rolling_std"])
axes[4].set_title(_col_to_workon+"_rolling_std")
plt.show()

# In[] Plot the results on top of each other
_cols_to_plot = [_col_to_workon, _col_to_workon+"_rolling_mean", _col_to_workon+"_rolling_median", _col_to_workon+"_rolling_var", _col_to_workon+"_rolling_std"]
df[_cols_to_plot].plot()
plt.legend(_cols_to_plot)
plt.title(f"Rolling method applied on {_col_to_workon} signal from gait data")
plt.show()

# In[] Other metrics
_cols_to_analyse = ["Gyr_Z", "FreeAcc_U"]
df_rolling_cov = df[_cols_to_analyse].copy().dropna().rolling(_windows_size).cov()

# In[] Plot the results
_indexes = range(1, len(df_rolling_cov), 2)

plt.clf()
plt.plot(df[_col_to_workon], label=_col_to_workon)
plt.plot(df["FreeAcc_U"], label="FreeAcc_U")
plt.plot(df_rolling_cov.iloc[_indexes]["Gyr_Z"].values, label="cov_results")
plt.legend()
plt.show()

# In[] Other Params
_cols_to_analyse = ["FreeAcc_mag", "Gyr_mag"]
df_rolling_cov_mag = df[_cols_to_analyse].copy().dropna().rolling(_windows_size).cov()

# In[]
_indexes = range(1, len(df_rolling_cov_mag), 2)
plt.clf()
plt.plot(df["Gyr_Z"], label="Gyr_Z")
plt.plot(df["Gyr_mag"], label="Gyr_mag")
plt.plot(df["FreeAcc_mag"], label="FreeAcc_mag")
plt.plot(df_rolling_cov_mag.iloc[_indexes]["FreeAcc_mag"].values, label="cov_results")
plt.legend()
plt.show()

