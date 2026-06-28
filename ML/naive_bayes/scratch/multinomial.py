

from collections import defaultdict
import math
import re
from typing import Dict, Iterable, List, Set, Tuple
from common import Message

def tokenize_without_set(text: str) -> List[str]:
  text = text.lower()
  all_words = re.findall('[0-9a-zа-яё]+', text)
  return all_words

class MultimodalNaiveBayesClassifier:
  def __init__(self, k: float = 0.5) -> None:
    self.tokens = set()
    self.k = k # сглаживание
    self.tokens_spam_count: Dict[str, int] = defaultdict(int)
    self.tokens_ham_count: Dict[str, int] = defaultdict(int)
    self.spam_messages = self.ham_messages = 0

  def _probabilities(self, token: str) -> Tuple[float, float]:
    spam_count = self.tokens_spam_count.get(token, 0)
    ham_count = self.tokens_ham_count.get(token, 0)

    # Сглаживание Лапласа
    spam_prob = (spam_count + self.k) / (self.total_spam_tokens + len(self.tokens) * self.k)
    ham_prob = (ham_count + self.k) / (self.total_ham_tokens + len(self.tokens) * self.k)
    return spam_prob, ham_prob

  def train(self, messages: Iterable[Message]) -> None:
    for message in messages:
      if message.is_spam:
        self.spam_messages += 1
      else:
        self.ham_messages += 1
      
      for token in tokenize_without_set(message.text):
        self.tokens.add(token)
        if message.is_spam:
          self.tokens_spam_count[token] += 1
        else:
          self.tokens_ham_count[token] += 1

    # нужно для мультимодального способа. 
    self.total_spam_tokens = sum(self.tokens_spam_count.values())
    self.total_ham_tokens = sum(self.tokens_ham_count.values())
    self.v_size = len(self.tokens)

  def predict(self, text: str) -> float:
    # Начинаем с логарифмов априорных вероятностей
    result_spam_prob = math.log(self.spam_messages / (self.spam_messages + self.ham_messages))
    result_ham_prob = math.log(self.ham_messages / (self.spam_messages + self.ham_messages))
    
    for token in tokenize_without_set(text):
      spam_prob, ham_prob = self._probabilities(token)
      # Использование логарифмов предотвращает зануление (Underflow)
      # при умножении малых вероятностей, заменяя умножение на сложение.
      result_spam_prob += math.log(spam_prob)
      result_ham_prob += math.log(ham_prob)

    diff = result_ham_prob - result_spam_prob

    # Если diff > 700, exp(diff) уйдет в бесконечность, значит вероятность спама стремится к 0
    if diff > 700:
        return 0.0
    # Если diff < -700, exp(diff) станет 0, значит вероятность спама стремится к 1
    elif diff < -700:
        return 1.0

    # Безопасное вычисление без  OverflowError или UnderflowError
    """
    Это сигмоида (или логистическая функция). 
    мы оперируем разностью весов, что снижает риск получить слишком огромное число,
    а так можно просто softmax:
    return math.exp(result_spam_prob) / (math.exp(result_spam_prob) + math.exp(result_ham_prob))
    """
    return 1 / (1 + math.exp(diff))

"""
Что можно улучшить в реализации ?
Добавить TF (Term Frequency): частота слова в конретном письме: 1 + log(50), где 50 - частота.
IDF (Inverse Document Frequency): редкость слова во всей базе.
Итоговый вес слова в документе: w = TF * IDF
"""
