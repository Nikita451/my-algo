from typing import List

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix
from ML.utils.split_utils import split_data
from ML.naive_bayes.scratch.common import Message

def run_sklearn_analysis(train_messages, test_messages):
    # 1. Готовим данные
    X_train = [m.text for m in train_messages]
    y_train = [1 if m.is_spam else 0 for m in train_messages]
    
    X_test = [m.text for m in test_messages]
    y_test = [1 if m.is_spam else 0 for m in test_messages]

    # 2. Векторизация текста
    #  Флаг binary=True заменяет подсчет количества слов на бинарное: 0 (нет) или 1 (есть)
    bernoulli_vectorizer = CountVectorizer(binary=True, lowercase=True, token_pattern=r'(?u)\b[0-9a-zа-яё]+\b')
    
    X_train_vectorized = bernoulli_vectorizer.fit_transform(X_train)
    # использует уже готовые правила/словари, чтобы превратить текст в числа
    X_test_vectorized = bernoulli_vectorizer.transform(X_test)
    bernoulli_model = BernoulliNB(alpha=1.0)

    bernoulli_model.fit(X_train_vectorized, y_train)
    predictions = bernoulli_model.predict(X_test_vectorized)
    
    # Вытаскиваем вероятности (у sklearn метод predict_proba возвращает матрицу [P_ham, P_spam])
    probabilities_spam = bernoulli_model.predict_proba(X_test_vectorized)[:, 1]

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

         Ham       0.87      1.00      0.93       686
        Spam       0.95      0.27      0.42       139

    accuracy                           0.88       825
   macro avg       0.91      0.64      0.68       825
weighted avg       0.88      0.88      0.84       825

=== Матрица ошибок (Confusion Matrix) ===
[[684   2]
 [101  38]]
"""