import numpy as np
import pandas as pd
import scipy.stats as stats
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

def movie_feat():
    '''
    create a function to add average in quantity of rated movieds
    :return:
    '''
    df = pd.read_csv('../dataset/movies_feat.csv', sep=',', index_col=0)
    df_ratings = pd.read_csv('../../../Dataset/ml-latest-small/ratings.csv')
    # Including average of ratings and count of movies rated (popularity)
    count_rated = df_ratings.groupby('movieId')['rating'].count()
    count_rated_df = pd.DataFrame({'count_rated': count_rated})

    avg_rating = df_ratings.groupby('movieId')['rating'].mean()
    avg_rating_df = pd.DataFrame({'avg_rating': avg_rating})

    df = pd.merge(df, avg_rating_df, left_on='movieId', right_index=True)
    df = pd.merge(df, count_rated_df, left_on='movieId', right_index=True)
    df = pd.merge(df, df_ratings, on='movieId', how='inner')

    df.drop(['userId', 'timestamp', 'title', 'movieId'], inplace=True, axis=1)
    X = df.drop('rating', axis=1)
    y = df['rating']
    X = X.apply(stats.zscore)

    X.to_csv('../dataset/X_feat.csv', sep=',')
    y.to_csv('../dataset/y.csv', sep=',')



class Process_text ():
    pass


def find_neigbohrs():
    df = pd.read_csv('../dataset/ml-latest-small/ratings.csv')
    df = df.pivot_table(index=df['userId'], columns='movieId', values='rating').fillna(0)

    # y = df.rating.values
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df)
    labels = kmeans.cluster_centers_
    print(labels[0:5])
    print(df[0:5])




find_neigbohrs()
