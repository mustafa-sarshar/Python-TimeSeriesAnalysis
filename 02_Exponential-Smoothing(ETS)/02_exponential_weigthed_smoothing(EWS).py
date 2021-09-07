"""
Exponential-weighted Smoothing
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

# In[] Data Analysis - Exponential Moving Window
_col_to_workon = "Gyr_Z"
_windows_size = 10
_alpha = 0.2
df_ewm = df.ewm(alpha=_alpha, adjust=False)
df[_col_to_workon+"_ewm_mean"] = df_ewm[_col_to_workon].mean()
df[_col_to_workon+"_ewm_vol"] = df_ewm[_col_to_workon].vol()
df[_col_to_workon+"_ewm_var"] = df_ewm[_col_to_workon].var()
df[_col_to_workon+"_ewm_std"] = df_ewm[_col_to_workon].std()

# In[] Plot the results
fig, axes = plt.subplots(5, 1)
axes[0].plot(df[_col_to_workon])
axes[0].set_title(_col_to_workon)
axes[1].plot(df[_col_to_workon+"_ewm_mean"])
axes[1].set_title(_col_to_workon+"_ewm_mean")
axes[2].plot(df[_col_to_workon+"_ewm_vol"])
axes[2].set_title(_col_to_workon+"_ewm_vol")
axes[3].plot(df[_col_to_workon+"_ewm_var"])
axes[3].set_title(_col_to_workon+"_ewm_var")
axes[4].plot(df[_col_to_workon+"_ewm_std"])
axes[4].set_title(_col_to_workon+"_ewm_std")
plt.show()

# In[] Plot the results all together
_cols_to_plot = [_col_to_workon, _col_to_workon+"_ewm_mean", _col_to_workon+"_ewm_vol", _col_to_workon+"_ewm_var", _col_to_workon+"_ewm_std"]
df[_cols_to_plot].plot()
plt.legend(_cols_to_plot)
plt.show()

# In[] Other Params
_cols_to_analyse = ["FreeAcc_mag", "Gyr_mag"]
df_rolling_cov_ewm = df[_cols_to_analyse].copy().dropna().ewm(_windows_size).cov()

# In[]
_indexes = range(1, len(df_rolling_cov_ewm), 2)
plt.clf()
plt.plot(df["Gyr_Z"], label="Gyr_Z")
plt.plot(df["Gyr_mag"], label="Gyr_mag")
plt.plot(df["FreeAcc_mag"], label="FreeAcc_mag")
plt.plot(df_rolling_cov_ewm.iloc[_indexes]["FreeAcc_mag"].values, label="cov_results")
plt.legend()
plt.show()