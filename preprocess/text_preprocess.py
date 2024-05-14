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
import os


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


class Preprocesstext:
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
        snow = SnowballStemmer('english')
        a = [snow.stem(i) for i in word_tokenize(self.text)]
        self.text = " ".join(a)


class PreprocessReviews:
    def __init__(self):
        home = False
        self.user = ''

        if home:
            self.user = 'eduardo'
        else:
            self.user = 'eduardoferreira'

        self.folder_path = ''

        self.train = "train"
        self.test = "test"
        self.url_pos = "urls_pos"
        self.url_neg = "urls_neg"

    def start(self):

        df_test_pos = self.preprocess(self.test, self.url_pos)
        df_test_neg = self.preprocess(self.test, self.url_neg)
        df_train_pos = self.preprocess(self.train, self.url_pos)
        df_train_neg = self.preprocess(self.train, self.url_neg)

        frames = [df_test_neg, df_test_pos, df_train_neg, df_train_pos]
        df_imdb = pd.concat(frames, ignore_index=True)
        df_imdb.to_csv('../dataset/processed/imdb_reviews.csv', sep=',', index=False)

    def preprocess(self, dataset, sentiment):
        self.folder_path = f'/home/{self.user}/Dataset/aclImdb/{dataset}/{sentiment.split("_")[1]}'
        dfs = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.txt'):
                # Extract id and rating from the filename
                id_rating = os.path.splitext(filename)[0]
                id_, rating = id_rating.split('_')
                # Read the content of the file
                file_path = os.path.join(self.folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    review = file.read()

                # Create a DataFrame for the current file
                df = pd.DataFrame({'id': [id_], 'rating': [rating], 'review': [review]})

                # Append the DataFrame to the list
                dfs.append(df)

        # Concatenate all DataFrames
        df = pd.concat(dfs)
        df['id'] = df['id'].astype(int)

        path = f'../../../Dataset/aclImdb/{dataset}/{sentiment}.txt'
        df_url_ = pd.read_csv(path, sep='\t', names=['url'])
        df_url = df_url_.url.str.split(r"http://www.imdb.com/title/|/usercomments", expand=True)
        df_url = df_url.reset_index(drop=True)
        df_url.rename(columns={1: 'imdbId'}, inplace=True)
        df_url = pd.DataFrame(df_url['imdbId'].apply(lambda x: x.split('tt')[-1]))
        df_res = pd.merge(df, df_url, left_on='id', right_index=True)

        if sentiment.split("_")[1] == "pos":
            df_res['sentiment'] = 1
        else:
            df_res['sentiment'] = 0

        return df_res
