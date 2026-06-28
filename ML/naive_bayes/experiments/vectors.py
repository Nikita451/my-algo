from typing import List

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# 1. Берем для примера всего 3 простых письма
sample_texts = [
    "Привет, купи акции",
    "Акция! Купи телефон",
    "Привет, как дела"
]

# 2. Инициализируем ваш векторизатор
demo_vectorizer = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b[0-9a-zа-яё]+\b')

# 3. Токенизируем и векторизуем тексты
X_demo = demo_vectorizer.fit_transform(sample_texts)

# 4. Извлекаем собранный словарь (уникальные токены)
# Они автоматически отсортированы по алфавиту
vocab = demo_vectorizer.get_feature_names_out()
print("=== СОБРАННЫЙ СЛОВАРЬ (ФИЧИ) ===")
print(vocab)
print("-" * 40)

# 5. Превращаем результат векторизации в обычную числовую матрицу
matrix = X_demo.toarray() # type: ignore

print("=== ВЕКТОРЫ ДЛЯ КАЖДОГО ПИСЬМА ===")
for text, vector in zip(sample_texts, matrix):
    print(f"Текст: '{text}'")
    print(f"Вектор чисел: {vector}\n")
