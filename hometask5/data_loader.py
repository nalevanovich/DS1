# data_loader.py

import os
import pandas as pd
from pathlib import Path

def load_data(path: Path, sample_frac: float = 1.0) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f'Файл не найден. Путь {path} неверный.')
        return None
    try:
        print('Файл успешно загружен')
        df = pd.read_csv(path)
        return df.sample(frac=sample_frac, random_state=42) if sample_frac < 1.0 else df
    except Exception as e:
        print(f'Возникла ошибка {e} при загрузке файла.')
        return None