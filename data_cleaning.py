import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def encode_dates(df: pd.DataFrame, col, replacement_field:str, time:bool=False, finetime:bool=False) -> pd.DataFrame:
    # from pandas docs
    # datetime_attr = ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'nanosecond', 'date', 'time', 'timetz', 'dayofyear', 'weekofyear', 'week', 'dayofweek', 'weekday', 'quarter', 'days_in_month', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end', 'is_leap_year']
    attr = date_attr = ['year', 'month', 'day']
    time_atr = ['hour', 'minute', 'second']
    finetime_attr = ['microsecond', 'nanosecond']

    if time:
        attr = attr + time_attr
    if finetime:
        attr = attr + finetime_attr

    for attr in date_attr:
        attr_col = [getattr(x, attr) for x in df[col]]
        df[replacement_field + attr] = pd.Series(attr_col)

    return df.drop(col, axis=1)

# from HOML Ch 4 (homl.info)
def plot_learning_curves(model, X, y, skip=100):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train), skip):
        try:
            model.fit(X_train[:m], y_train[:m])
        except:
            continue
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), 'b-', linewidth=2, label="val")
