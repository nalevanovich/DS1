#data_loader.py
import pandas as pd
import os

#Loading .csv
def load_data_csv(path):
    cols = ["target", "ids", "date", "flag", "user", "text"]
    if not os.path.exists(path):
        print(f'Файл не найден, путь {path} неверный.')
        return None
    try:
        print('Файл успешно загружен')
        return pd.read_csv('training.1600000.processed.noemoticon.csv', 
                 encoding='latin-1', 
                 names=cols)
    except Exception as e:
        print(f'Возникла ошибка {e} при загрузке файла')
        return None

def describe_data(df):
    print(df.describe(include = "all"))
    print(df.shape)
    print(df.info())
    print(f'Missing data:\n{df.isnull().sum()}')
    print(f'Список столбцов с пропущенными значениями: {df.columns[df.isnull().any()].tolist()}')