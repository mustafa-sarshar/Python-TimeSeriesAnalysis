"""
Augmented Dickey-Fuller Test (ADF Test)
"""
# In[] Libs
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller

# In[] define custom methods
def adf(data, alpha_level):
    res = adfuller(data)
    print(f"Test-Statistic: {res[0]:0.2f}")
    print(f"P-Value: {res[1]:0.2f}")
    if res[1] < alpha_level:
        print("Stationary")
    else:
        print("Non-Stationary")

# In[] Generate some random datasets
n_samples = 200

data_stationary_1 = np.random.normal(loc=0, scale=10, size=n_samples)
plt.plot(data_stationary_1, label="Normal distribution")
adf(data=data_stationary_1, alpha_level=0.05)

data_stationary_2 = np.random.gamma(shape=1, scale=1, size=n_samples)
plt.plot(data_stationary_2, label="Gamma distribution")
adf(data=data_stationary_2, alpha_level=0.05)

plt.legend()
plt.show()
