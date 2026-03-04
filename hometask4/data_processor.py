# data_processor.py
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

eyes = r'[:;=]'
nose = r'-?'
happy_pattern = rf'({eyes}{nose}[D\)P\]])|(\){nose}{eyes})'
sad_pattern = rf'({eyes}{nose}[\(\[S\|])|([\[\(\|SD]{nose}{eyes})'

happy_re = re.compile(happy_pattern)
sad_re = re.compile(sad_pattern)


def get_features(text: str) -> tuple[int, int, int, int, int, int]:
    """
    Извлечь признаки из сырого (неочищенного) текста.
    Необходимо вызвать перед clean_text(), чтобы сохранить заглавные буквы, пунктуацию и эмодзи.
    """
    if not isinstance(text, str):
        return 0, 0, 0, 0, 0, 0

    char_count = len(text)

    words = text.split()
    word_count = len(words)

    is_caps = len([w for w in words if w.isupper() and len(w) > 2])

    excl_count = text.count('!')

    h = 1 if happy_re.search(text) else 0
    s = 1 if sad_re.search(text) else 0

    return h, s, char_count, word_count, is_caps, excl_count


def clean_text(text: str) -> str:
    """
    Очистить и нормализовать текст для подачи на вход модели.
    Вызовите get_features() для сырого текста ПЕРЕД этой функцией.
    """
    if not isinstance(text, str):  # ← Explicit guard instead of silent failure
        return ""

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]

    return " ".join(cleaned_words)