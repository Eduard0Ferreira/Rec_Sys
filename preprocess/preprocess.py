import pandas as pd
import numpy as np
import scipy.stats as stats

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


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

    def dataset(self, batch_size) -> (DataLoader, DataLoader):
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
        X = list(zip(matrix.userId.values, matrix.movieId.values))
        y = matrix.rating.values
        data_t = torch.tensor(X)
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

    def get_movie(self, id_movie: int) -> dict:
        return self.df_movie.iloc[id_movie]

    def num_user_item(self):
        return self.df_ratings['userId'].nunique(), self.df_ratings['movieId'].nunique()

    def get_matrix(self) -> pd.DataFrame:
        return self.df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    def get_ids(self) -> (pd.Series, pd.Series):
        print(type(self.df_ratings['userId'].unique()))
        return self.df_ratings['userId'].unique(), self.df_ratings['movieId'].unique()






