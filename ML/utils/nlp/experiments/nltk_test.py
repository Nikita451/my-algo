import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import WordNetLemmatizer

# Загружаем токенизатор NLTK (Punkt)
nltk.download("punkt_tab")

# word_tokenize
text = "Do you want to enroll in the finest courses on machine learning?? \
I studied there, it was amazing!!"
tokenized_text = word_tokenize(text)
print(f"Source text: {text}")
print(f"Tokenized text: {tokenized_text}")


# Lowercased
lowercased_text = [word.lower() for word in tokenized_text if word.isalpha()]
print(f"Tokenized text: {tokenized_text}")
print(f"Lowercased text: {lowercased_text}")


# СТОП СЛОВА
nltk.download("stopwords")
print(f"Stopwords in English: {stopwords.words('english')}")

STOPWORDS = stopwords.words("english")
text_without_stopwords = [word for word in lowercased_text if word not in STOPWORDS]
print(f"Lowercased text: {lowercased_text}")
print(f"Text without stopwords: {text_without_stopwords}")


"""
Стемминг — эвристический процесс нахождения основы для заданного слова. 
Он осуществляется на основе списка наиболее распространенных окончаний и 
словообразовательных суффиксов.
"""
stemmer = PorterStemmer()
stemmed_text = [stemmer.stem(word) for word in text_without_stopwords]
print(f"Text without stopwords: {text_without_stopwords}")
print(f"Stemmed text: {stemmed_text}")


"""
Лемматизация — процесс приведения словоформы к лемме, то есть её нормальной (словарной) 
форме. Для его применения необходима информация о части речи и морфологической форме 
анализируемого слова, что требует анализа контекста.

По умолчанию параметр pos (part of speech) лемматизатора WordNetLemmatizer имеет значение 
"n" (noun). Другие возможные значения: "v" (verb) для глаголов, "a" (adjective) для 
прилагательных и "r" (adverb) для наречий.
"""
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
lemmatized_text = [lemmatizer.lemmatize(word) for word in text_without_stopwords]
print(f"Text without stopwords: {text_without_stopwords}")
print(f"Lemmatized text: {lemmatized_text}")
print(f'Lemma for word "studied" is {lemmatizer.lemmatize("studied", "v")}.')
print(f'Lemma for word "finest" is {lemmatizer.lemmatize("finest", "a")}.')

# определяем часть речи - pos_tag
nltk.download("averaged_perceptron_tagger_eng")
print(nltk.pos_tag(text_without_stopwords))

# приводим части речи к единому формату.
def pos_mapping(text):
    word_tag_array = []
    for word_tag in nltk.pos_tag(text):
        tag = word_tag[1][0] # ('want', 'JJ'), -> J
        tag_dict = {"J": "a", "N": "n", "V": "v", "R": "r"}
        word_tag_array.append((word_tag[0], tag_dict[tag]))
    return word_tag_array

print(pos_mapping(text_without_stopwords))

# лемматизацию всех частей речи
lemmatized_text = [
    lemmatizer.lemmatize(word, tag) for word, tag in pos_mapping(text_without_stopwords)
]
print(f"Text without stopwords: {text_without_stopwords}")
print(f"Lemmatized text: {lemmatized_text}")
