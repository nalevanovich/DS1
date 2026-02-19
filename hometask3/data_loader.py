#data_loader.py
import pandas as pd
import os
import requests

#Loading .csv
def load_data_csv(path):
    if not os.path.exists(path):
        print(f'Файл не найден, путь {path} неверный.')
        return None
    try:
        print('Файл успешно загружен')
        return pd.read_csv(path)
    except Exception as e:
        print(f'Возникла ошибка {e} при загрузке файла')
        return None

#Loading .json
def load_data_json(path):
    if not path.os.exist(path):
        print(f'Файл не найден, путь {path} неверный.')
        return None
    try:
        print('Файл успешно загружен')
        return pd.read_json(path)
    except Exception as e:
        print(f'Возникла ошибка {e} при загрузке файла')
        return None

#Loading API
def load_data_api(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к API: {e}")
        return None