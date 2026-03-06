# src/utils.py
import pandas as pd
import numpy as np

def reduce_memory_usage(df):
    """ Reduce memory usage of a DataFrame by downcasting numerical columns. """
    print(f"Initial memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns:
        col_type = df[col].dtype
        if str(col_type)[:5] == 'float':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
        elif str(col_type)[:3] == 'int':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization: {end_mem:.2f} MB")
    if start_mem > 0: # Avoid division by zero if start_mem is 0
        print(f"Decreased by {(start_mem - end_mem) * 100 / start_mem:.1f}%")
    return df