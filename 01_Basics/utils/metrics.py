def metrics_sse(y_true, y_pred):
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return (y_true - y_pred).dot(y_true - y_pred)

def metrics_mse(y_true, y_pred):
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return (y_true - y_pred).dot(y_true - y_pred) / len(y_true)

def metrics_rmse(y_true, y_pred):
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt((y_true - y_pred).dot(y_true - y_pred) / len(y_true))

def metrics_mae(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)

def metrics_mape(y_true, y_pred):
    import numpy as np
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def metrics_smape(y_true, y_pred):
    import numpy as np
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = numerator / denominator
    return ratio.mean()

def metrics_r2_score(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)

