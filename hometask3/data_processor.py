#data_processor.py
import pandas as pd

#Describe data
def data_discribe(df):
    print(df.describe(include = "all"))
    print(df.shape)
    print(df.info())
    print(f'Missing data:\n{df.isnull().sum()}')
    print(f'Список столбцов с пропущенными значениями: {df.columns[df.isnull().any()].tolist()}')

#Filling missing data with mean
def data_fill_mean(df):
    df = df.copy()
    numeric_cols_nan = df.select_dtypes(include=['number']).columns[df.select_dtypes(include=['number']).isnull().any()].tolist()
    for col in numeric_cols_nan:
        if col in df.columns:
            mean = df[col].mean()
            df[col] = df[col].fillna(mean)
    return df

#Filling missing data with median
def data_fill_median(df):
    df = df.copy()
    numeric_cols_nan = df.select_dtypes(include=['number']).columns[df.select_dtypes(include=['number']).isnull().any()].tolist()
    for col in numeric_cols_nan:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)
    return df

#Filling missing data with mode
def data_fill_freq(df):
    df = df.copy()
    list_cols_nan = df.columns[df.isnull().any()].tolist()
    for col in list_cols_nan:
        if col in df.columns:
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
    return df

#Filling missing with 'Unknown'
def data_fill_unkn(df):
    df = df.copy()
    list_cols_nan = df.columns[df.isnull().any()].tolist()
    for col in list_cols_nan:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    return df