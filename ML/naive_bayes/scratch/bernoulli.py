from collections import defaultdict
import math
import re
from typing import Dict, Iterable, List, Set, Tuple
from common import Message


def tokenize(text: str) -> Set[str]:
  text = text.lower()
  all_words = re.findall('[0-9a-zа-яё]+', text)
  return set(all_words)

class BernullyNaiveBayesClassifier:

  def __init__(self, k: float = 0.5) -> None:
    self.tokens = set()
    self.k = k # сглаживание
    self.tokens_spam_count: Dict[str, int] = defaultdict(int)
    self.tokens_ham_count: Dict[str, int] = defaultdict(int)
    self.spam_messages = self.ham_messages = 0


  def _probabilities(self, token: str) -> Tuple[float, float]:
    spam_count = self.tokens_spam_count.get(token, 0)
    ham_count = self.tokens_ham_count.get(token, 0)

    spam_prob = (spam_count + self.k) / (self.spam_messages + 2 * self.k)
    ham_prob = (ham_count + self.k) / (self.ham_messages + 2 * self.k)

    return spam_prob, ham_prob

  def train(self, messages: Iterable[Message]) -> None:
    for message in messages:
      if message.is_spam:
        self.spam_messages += 1
      else:
        self.ham_messages += 1
      
      for token in tokenize(message.text):
        self.tokens.add(token)
        if message.is_spam:
          self.tokens_spam_count[token] += 1
        else:
          self.tokens_ham_count[token] += 1

    # Вместо обхода всего словаря, мы заранее суммируем логарифмы (1 - p) для ВСЕХ слов.
    # Это "базовый вес" сообщения, в котором нет ни одного слова из словаря.
    self.base_log_spam = 0.0
    self.base_log_ham = 0.0
    for token in self.tokens:
        p_s, p_h = self._probabilities(token)
        self.base_log_spam += math.log(1 - p_s)
        self.base_log_ham += math.log(1 - p_h)

  def predict(self, text: str) -> float:
    # 1. Считаем априорные вероятности
    log_spam_prob = math.log(self.spam_messages / (self.spam_messages + self.ham_messages)) + self.base_log_spam
    log_ham_prob = math.log(self.ham_messages / (self.spam_messages + self.ham_messages)) + self.base_log_ham

    # Теперь смотрим, какие слова НА САМОМ ДЕЛЕ есть в сообщении
    # Для них мы должны ЗАМЕНИТЬ "вес отсутствия" на "вес наличия"
    tokens_in_message = tokenize(text) # Здесь set() обязателен для Бернулли
    
    for token in tokens_in_message:
        if token in self.tokens:
            p_s, p_h = self._probabilities(token)
            
            # Корректировка: вычитаем добавленный ранее log(1-p) и прибавляем log(p)
            log_spam_prob += math.log(p_s) - math.log(1 - p_s)
            log_ham_prob += math.log(p_h) - math.log(1 - p_h)

    # Защита от переполнения
    diff = log_ham_prob - log_spam_prob
    if diff > 700:
        return 0.0
    elif diff < -700:
        return 1.0

    # Устойчивая логистическая функция (сигмоида)
    return 1 / (1 + math.exp(diff))

