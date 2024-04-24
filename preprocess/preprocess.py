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

class MF(nn.Module):

    def __init__(self, num_users, num_items, emb_dim, init):
        super().__init__()
        self.user_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=emb_dim)
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=emb_dim)

        self.fc1 = nn.Linear(emb_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # add bias
        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(num_items))
        self.offset = nn.Parameter(torch.zeros(1))

        if init:
            self.user_emb.weight.data.uniform_(0., 0.5)
            self.item_emb.weight.data.uniform_(0., 0.5)

    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        element_product = (user_emb * item_emb).sum(1)

        x = self.relu(self.fc1(element_product))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        user_b = self.user_bias[user]
        item_b = self.item_bias[item]
        # element_product += user_b + item_b + self.offset

        return x + user_b + item_b + self.offset


n_users = len(df_user_item.userId.unique())
n_items = len(df_user_item.movieId.unique())
mf_model = MF(n_users, n_items, emb_dim=32, init=True)
mf_model.to(device)
print(mf_model)