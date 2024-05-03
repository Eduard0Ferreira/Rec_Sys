import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


class RecSys:

    def __init__(self, model, user_item_matrix, user_ids, item_ids, movies):
        self.model = model
        self.user_item_matrix = user_item_matrix
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.movies = movies

    def make_recommendations(self, user_id, n=10):
        # get predictions from trained model
        user_input = torch.full((len(self.item_ids),), user_id, dtype=torch.long)
        item_input = torch.tensor(self.item_ids, dtype=torch.long)
        predictions = self.model(user_input, item_input)

        # Exclude items the user has already rated
        user_ratings = self.user_item_matrix[user_id]

        for i, rating in enumerate(user_ratings):
            if rating != 0:
                predictions[i] = float('-1')  # set to negative evaluation movies watched

        # Get indices of top-N predictions
        top_n_indices = torch.topk(predictions, n).indices.cpu().numpy()

        # Prepare recommendations
        recommendations = []
        for i in top_n_indices:
            movie_id = self.item_ids[i]
            title = self.movies['title'].iloc[movie_id]
            predicted_rating = predictions[i]
            recommendations.append({'movie_name': title, 'predicted_rating': predicted_rating})

        return recommendations
