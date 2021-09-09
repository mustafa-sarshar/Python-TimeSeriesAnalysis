"""
Temporary Code for testing and discorvering
"""
# In[] Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# In[] Datasets
_dataset_dir = "GaitData"
_dataset_file = "03_LeftAnkle.csv"
df = pd.read_csv(f"..\\datasets\\{_dataset_dir}\\{_dataset_file}", skiprows=12)
_cols_to_drop = ["PacketCounter"]  
df.drop(columns=_cols_to_drop, inplace=True)

# In[] Plot the dataset
_cols_to_plot = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
plt.clf()
plt.plot(df[_cols_to_plot])
plt.legend(_cols_to_plot)
plt.show()

# In[]
df["Acc_all_mul"] = df["FreeAcc_E"] * df["FreeAcc_N"] * df["FreeAcc_U"]
df["Gyr_all_mul"] = df["Gyr_X"] * df["Gyr_Y"] * df["Gyr_Z"]
df["Acc_all_sum"] = df["FreeAcc_E"] + df["FreeAcc_N"] + df["FreeAcc_U"]
df["Gyr_all_sum"] = df["Gyr_X"] + df["Gyr_Y"] + df["Gyr_Z"]
df["Acc_all_mag"] = (df["FreeAcc_E"]**2 + df["FreeAcc_N"]**2 + df["FreeAcc_U"]**2)**.5
df["Gyr_all_mag"] = (df["Gyr_X"]**2 + df["Gyr_Y"]**2 + df["Gyr_Z"]**2)**.5
plt.clf()
# plt.plot(MaxAbsScaler().fit_transform(df["Acc_all_sum"].values.reshape(-1, 1)), label="Acc_all_sum")
# plt.plot(MaxAbsScaler().fit_transform(df["Gyr_all_sum"].values.reshape(-1, 1)), label="Gyr_all_sum")
plt.plot(MaxAbsScaler().fit_transform(df["Acc_all_mag"].values.reshape(-1, 1)), label="Acc_all_mag")
plt.plot(MaxAbsScaler().fit_transform(df["Gyr_all_mag"].values.reshape(-1, 1)), label="Gyr_all_mag")
plt.plot(-MaxAbsScaler().fit_transform(df["Gyr_Z"].values.reshape(-1, 1)), label="Gyr_Z")
plt.legend()
plt.show()

# In[]
df["Gyr_all_mag_log"] = np.log(df["Gyr_all_mag"] + 1)
df["Gyr_all_mag_log_diff_1"] = df["Gyr_all_mag_log"].diff()
plt.clf()
plt.plot(df["Gyr_all_mag_log"], label="Gyr_all_mag_log")
plt.plot(df["Gyr_all_mag_log_diff_1"], label="Gyr_all_mag_log_diff_1")
plt.legend()
plt.show()

# In[] Plot the PACF and ACF functions to determine p and q hyperparameters for ARIMA model
_cols_to_workon = ["Gyr_all_mag", "Gyr_all_mag_log", "Gyr_all_mag_log_diff_1"]
fig, axes = plt.subplots(2, 3)
for jj in range(axes.shape[1]):
    plot_pacf(x=df[_cols_to_workon[jj]], ax=axes[0][jj])
    plot_acf(x=df[_cols_to_workon[jj]], ax=axes[1][jj])
    
    axes[0][jj].set_title("[PACF]  "+_cols_to_workon[jj])
    axes[1][jj].set_title("[ACF]  "+_cols_to_workon[jj])
    axes[0][jj].set_ylim(-1.1, 1.1)
    axes[1][jj].set_ylim(-1.1, 1.1)
plt.show()

# In[] Finish