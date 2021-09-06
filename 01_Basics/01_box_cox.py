"""
Time Series Transformation by using Boxcox
"""
# In[] Libs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox

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
scaler = MinMaxScaler(feature_range=(0.01, 1.01)) # Data must contain only positive values, therefore we scale the data from 0.01 to 1.01.
# df[_cols_to_workon] = np.log(df[_cols_to_workon]) # uncomment to apply log transform
df[_cols_to_workon+"_scaled"] = scaler.fit_transform(X=df[_cols_to_workon].values.reshape(-1, 1))
data, lam = boxcox(df[_cols_to_workon+"_scaled"]) 
df[_cols_to_workon+"_boxcox"] = data

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
