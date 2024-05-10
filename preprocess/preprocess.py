import pandas as pd
import numpy as np
import scipy.stats as stats

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import os


#
# def movie_feat():
#     '''
#     create a function to add average in quantity of rated movieds
#     :return:
#     '''
#     df = pd.read_csv('../dataset/movies_feat.csv', sep=',', index_col=0)
#     df_ratings = pd.read_csv('../../../Dataset/ml-latest-small/ratings.csv')
#     # Including average of ratings and count of movies rated (popularity)
#     count_rated = df_ratings.groupby('movieId')['rating'].count()
#     count_rated_df = pd.DataFrame({'count_rated': count_rated})
#
#     avg_rating = df_ratings.groupby('movieId')['rating'].mean()
#     avg_rating_df = pd.DataFrame({'avg_rating': avg_rating})
#
#     df = pd.merge(df, avg_rating_df, left_on='movieId', right_index=True)
#     df = pd.merge(df, count_rated_df, left_on='movieId', right_index=True)
#     df = pd.merge(df, df_ratings, on='movieId', how='inner')
#
#     df.drop(['userId', 'timestamp', 'title', 'movieId'], inplace=True, axis=1)
#     X = df.drop('rating', axis=1)
#     y = df['rating']
#     X = X.apply(stats.zscore)
#
#     X.to_csv('../dataset/X_feat.csv', sep=',')
#     y.to_csv('../dataset/y.csv', sep=',')

class MovieLens:
    def __init__(self):
        self.df_ratings = pd.read_csv('../dataset/ml-latest-small/ratings.csv')
        self.df_movie = pd.read_csv('../dataset/ml-latest-small/movies.csv')

    def dataset(self, batch_size: int) -> (DataLoader, DataLoader):
        """
        Get dataframe to process, apply the encoder for index the user and movie
        :return: X and y in torch tensor format
        """
        d = defaultdict(LabelEncoder)
        cols_cat = ['userId', 'movieId']
        for c in cols_cat:
            d[c].fit(self.df_ratings[c].unique())
            self.df_ratings[c] = d[c].transform(self.df_ratings[c])

        matrix = self.df_ratings[['userId', 'movieId', 'rating']]
        x = list(zip(matrix.userId.values, matrix.movieId.values))
        y = matrix.rating.values

        data_t = torch.tensor(x)
        labels = torch.tensor(y)

        # split the data
        train_data, test_data, train_labels, test_labels = train_test_split(data_t, labels, test_size=.1)

        # convert to a pytorch
        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)

        # train and test dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=test_dataset.tensors[0].shape[0])

        return train_loader, test_loader

    def dataset_encoder(self, batch_size: int) -> DataLoader:
        matrix = torch.tensor(self.get_matrix().values)

        # convert to a pytorch
        train_dataset = TensorDataset(matrix)

        # train dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        return train_loader

    def get_movies(self) -> pd.DataFrame:
        return self.df_movie[['movieId', 'title']]

    def num_user_item(self):
        return self.df_ratings['userId'].nunique(), self.df_ratings['movieId'].nunique()

    def get_matrix(self) -> pd.DataFrame:
        return self.df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    def get_ids(self) -> (pd.Series, pd.Series):
        # print(type(self.df_ratings['userId'].unique()))
        return self.df_ratings['userId'].unique(), self.df_ratings['movieId'].unique()


class Tags_Movielens:

    def __init__(self):
        self.df_tags = pd.read_csv('../dataset/ml-latest-small/tags.csv', sep=',')
        self.df_links = pd.read_csv('../dataset/ml-latest-small/links.csv', sep=',')

    def preprocess(self):
        tags = pd.merge(self.df_tags, self.df_links, on='movieId', how='inner')
        tags.drop(['tmdbId', 'timestamp'], inplace=True, axis=1)
        # group tags by imdbId
        tags['imdbId'] = tags['imdbId'].astype(str)
        tags['tag'] = tags['tag'].astype(str)
        df_group_tag = tags.groupby('imdbId')['tag'].apply(lambda x: ", ".join(x)).reset_index()
        df_group_tag.to_csv('../dataset/processed/tags.csv', index=False)


class ProcessReviews:
    def __init__(self):
        home = True
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
                id_rating = os.path.splitext(filedescribename)[0]
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
