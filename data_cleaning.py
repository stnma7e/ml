import datetime
import pandas as pd

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
