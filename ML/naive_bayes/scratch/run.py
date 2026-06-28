from io import BytesIO
from typing import List
from ML.naive_bayes.scratch.bernoulli import BernullyNaiveBayesClassifier
from ML.naive_bayes.scratch.multinomial import MultimodalNaiveBayesClassifier
from ML.utils.split_utils import split_data
import requests
import tarfile
from common import Message
from collections import Counter


BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
FILES = ["20021010_easy_ham.tar.bz2",
"20021010_hard_ham.tar.bz2",
"20021010_spam.tar.bz2"]
# В этих папках данные окажутся после распаковки:
# /spam, /easy_ham и /hard_ham.
OUTPUT_DIR = 'spam_data'
# for filename in FILES:
#   content = requests.get(f"{BASE_URL}/{filename}").content
#   # Обернуть байты в памяти, чтобы использовать их как "файл"
#   fin = BytesIO(content)
#   # И извлечь все файлы в указанный выходной каталог.
#   with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
#     tf.extractall(OUTPUT_DIR)


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

model = MultimodalNaiveBayesClassifier() # Counter({(False, False): 585, (True, True): 119, (False, True): 101, (True, False): 20})
# model = BernullyNaiveBayesClassifier() #

model.train(train_messages)
predictions = [(message, model.predict(message.text)) for message in test_messages]

confusion_matrix = Counter((message.is_spam, spam_probability > 0.5) for message, spam_probability in predictions)
print(confusion_matrix) 
