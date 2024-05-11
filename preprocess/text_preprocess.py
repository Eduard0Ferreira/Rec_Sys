import pandas as pd
import numpy as np

# for text pre-processing
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
# for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# for word embedding
import gensim
from gensim.models import Word2Vec


def get_wordnet_pos(word):
    if word.startswith('J'):
        return wordnet.ADJ
    elif word.startswith('V'):
        return wordnet.VERB
    elif word.startswith('N'):
        return wordnet.NOUN
    elif word.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


class Preprocessor:
    def __init__(self):
        super().__init__()
        self.text = ''

    def start(self, sentence: str) -> str:
        self.text = sentence
        self.preprocess()
        self.stopword()
        self.stemming()
        self.lemmatizer()

        return self.text

    def preprocess(self):
        self.text = self.text.lower()
        self.text = self.text.strip()
        self.text = re.compile('<.*?>').sub('', self.text)
        self.text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', self.text)
        self.text = re.sub('\s+', ' ', self.text)
        self.text = re.sub(r'\[[0-9]*\]', ' ', self.text)
        self.text = re.sub(r'[^\w\s]', '', str(self.text).lower().strip())
        self.text = re.sub(r'\d', ' ', self.text)
        self.text = re.sub(r'\s+', ' ', self.text)

    # STOPWORD REMOVAL
    def stopword(self):
        a = [i for i in self.text.split() if i not in stopwords.words('english')]
        self.text = ' '.join(a)

    def lemmatizer(self):
        wl = WordNetLemmatizer()

        word_pos_tags = nltk.pos_tag(word_tokenize(self.text))  # Get position tags
        a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in
             enumerate(word_pos_tags)]  # Map the position tag and lemmatize the word/token
        self.text = " ".join(a)

    def stemming(self):
        print('stemming', self.text)
        snow = SnowballStemmer('english')
        a = [snow.stem(i) for i in word_tokenize(self.text)]
        self.text = " ".join(a)
