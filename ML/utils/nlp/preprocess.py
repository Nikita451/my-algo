import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer, WordNetLemmatizer

nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")


def text_preprocessing(text):
    tokenized_text = word_tokenize(text)
    STOPWORDS = stopwords.words("english")
    text_without_stopwords = [
        word.lower()
        for word in tokenized_text
        if word.isalpha() and word not in STOPWORDS
    ]
    stemmer = PorterStemmer()
    stemmed_text = [stemmer.stem(word) for word in text_without_stopwords]
    return " ".join(stemmed_text)

def text_preprocessing_lemma(text):
    tokenized_text = word_tokenize(text)
    STOPWORDS = stopwords.words("english")
    text_without_stopwords = [
        word.lower()
        for word in tokenized_text
        if word.isalpha() and word not in STOPWORDS
    ]
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [
        lemmatizer.lemmatize(word, tag) for word, tag in pos_mapping(text_without_stopwords)
    ]
    return " ".join(lemmatized_text)


def pos_mapping(text):
    word_tag_array = []
    for word_tag in nltk.pos_tag(text):
        tag = word_tag[1][0] # ('want', 'JJ'), -> J
        tag_dict = {"J": "a", "N": "n", "V": "v", "R": "r"}
        word_tag_array.append((word_tag[0], tag_dict[tag]))
    return word_tag_array

