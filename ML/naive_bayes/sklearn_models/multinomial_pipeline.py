from typing import List

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from scratch.common import Message
from ML.utils.split_utils import split_data

def run_sklearn_analysis(train_messages, test_messages):
    # 1. Готовим данные
    X_train = [m.text for m in train_messages]
    y_train = [1 if m.is_spam else 0 for m in train_messages]
    
    X_test = [m.text for m in test_messages]
    y_test = [1 if m.is_spam else 0 for m in test_messages]

    # 2. Векторизация текста
    vectorizer = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b[0-9a-zа-яё]+\b')
    
    X_train_vectorized = vectorizer.fit_transform(X_train)
    # использует уже готовые правила/словари, чтобы превратить текст в числа
    X_test_vectorized = vectorizer.transform(X_test)

    # 3. Настройка GridSearchCV для поиска лучшего сглаживания (вашего k, в sklearn это alpha)
    # Проверим разные значения alpha, чтобы найти идеальное
    param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}
    nb_model = MultinomialNB()
    
    # 2. ИНИЦИАЛИЗИРУЕМ СЕТОЧНЫЙ ПОИСК (GridSearchCV)
    # Этот инструмент автоматически найдет лучшую модель из сетки выше.
    grid_search = GridSearchCV(
        estimator=nb_model, # Какую базовую модель обучаем
        param_grid=param_grid,      # Набор параметров для перебора
        cv=5,                       # Кросс-валидация: разбиваем трейн на 5 частей (блоков),
                                    # обучаемся на 4-х, тестируем на 1-й. Повторяем 5 раз.
        scoring='f1',               # Метрика качества. Для спама F1-score лучше балансирует 
                                    # между точностью (Precision) и полнотой (Recall)
        n_jobs=-1                   # Использовать все ядра процессора Mac для параллельного ускорения
    )

    # 3. ЗАПУСКАЕМ ПОИСК И ОБУЧЕНИЕ
    # Под капотом запустится цикл: 4 параметра * 5 блоков CV = 20 отдельных обучений!
    grid_search.fit(X_train_vectorized, y_train)

    print(f"Лучшее значение параметра alpha (k): {grid_search.best_params_['alpha']}")
    
    # 4. Предсказание на лучшей модели
    best_model = grid_search.best_estimator_
    # возвращает массив, состоящий строго из 0 (Ham) или 1 (Spam).
    predictions = best_model.predict(X_test_vectorized)
    
    # Вытаскиваем вероятности (у sklearn метод predict_proba возвращает матрицу [P_ham, P_spam])
    probabilities_spam = best_model.predict_proba(X_test_vectorized)[:, 1]

    # 5. вывод метрик
    print("\n=== Отчет по метрикам (Classification Report) ===")
    print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))
    
    print("=== Матрица ошибок (Confusion Matrix) ===")
    print(confusion_matrix(y_test, predictions))
    
    return probabilities_spam


# ПОДГОТОВКА ДАННЫХ
import glob, re, random
path = 'spam_data/*/*'
data: List[Message] = []

for filename in glob.glob(path):
  is_spam = "ham" not in filename
  # В письмах имеются несколько мусорных символов;
  # параметр errors='ignore' пропускает их исключения
  with open(filename, errors='ignore') as email_file:
    for line in email_file:
      if line.startswith("Subject:"):
        subject = line.lstrip( "Subject: ")
        data.append(Message(subject, is_spam))
        break

random.seed(0)
train_messages, test_messages =  split_data(data, 0.75) 

run_sklearn_analysis(train_messages, test_messages)



"""
=== Отчет по метрикам (Classification Report) ===
              precision    recall  f1-score   support

         Ham       0.92      0.98      0.95       686
        Spam       0.84      0.58      0.69       139

    accuracy                           0.91       825
   macro avg       0.88      0.78      0.82       825
weighted avg       0.91      0.91      0.90       825

=== Матрица ошибок (Confusion Matrix) ===
[[671  15]
"""