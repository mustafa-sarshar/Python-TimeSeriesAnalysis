"""
Naive Forecast
"""
# In[] Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import metrics

# In[] Datasets
_dataset_dir = "GaitData"
_dataset_file = "01_LeftAnkle.csv"
df = pd.read_csv(f"..\\Datasets\\{_dataset_dir}\\{_dataset_file}", skiprows=12)
df.drop(columns=["PacketCounter", "StatusWord", "ClippingFlags", "RSSI"], inplace=True)

# In[] Plot the dataset
_cols_to_plot = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
plt.clf()
plt.plot(df[_cols_to_plot])
plt.legend(_cols_to_plot)
plt.show()

# In[] Data Analysis
_cols_to_workon = "Gyr_Z"
df[_cols_to_workon+"_naive_forecast"] = df[_cols_to_workon].shif(1)
y_true = df.iloc[1:][_cols_to_workon]
y_pred = df.iloc[1:][_cols_to_workon+"_naive_forecast"]

# In[] Plot the results
_cols_to_plot = [_cols_to_workon+"_scaled", _cols_to_workon+"_boxcox"]
fig, axes = plt.subplots(3, 1)
axes[0].plot(df[_cols_to_plot])
axes[0].legend(_cols_to_plot)
axes[1].hist(df[_cols_to_plot[0]])
axes[1].legend([_cols_to_workon+"_scaled"+"_hist"])
axes[2].hist(df[_cols_to_plot[1]], label=_cols_to_workon+"_boxcox"+"_hist")
axes[2].legend([_cols_to_workon+"_boxcox"+"_hist"])
plt.show()

# In[]
