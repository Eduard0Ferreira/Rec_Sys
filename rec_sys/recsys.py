import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


class RecSys:

    def __init__(self, model, user_item_matrix, user_ids, item_ids, device):
        self.model = model
        self.user_item_matrix = user_item_matrix
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.device = device

    def train_model(self):
        pass

    def test_model(self):
        pass

    def make_recommendations(self, user_id, n=10):
        user_ratings = self.user_item_matrix.iloc[user_id]

        # user_ratings_tensor = torch.tensor(user_ratings, dtype=torch.float32).to(self.device)

        # Get predictions for all items
        all_predictions = self.model(torch.full((len(self.item_ids),), user_id).to(self.device),
                                     torch.tensor(self.item_ids, dtype=torch.long).to(self.device))

        # Exclude items the user has already rated
        all_predictions[user_ratings != 0] = 0

        # Get indices of top-N predictions
        top_n_indices = torch.topk(all_predictions, n).indices.cpu().numpy()

        # Map indices to item IDs
        top_n_items = [self.item_ids[i] for i in top_n_indices]
        top_n_values = all_predictions[top_n_indices].detach().cpu().numpy()

        return top_n_items, top_n_values

    def make_rec(self, model, user_id, item_ids, movie_names, user_item_matrix):
        # Prepare inputs
        user_input = torch.full((len(item_ids),), user_id, dtype=torch.long)
        item_input = torch.tensor(item_ids, dtype=torch.long)

        # Pass inputs through the model
        predictions = model(user_input, item_input)

        # Convert predictions to numpy array
        predictions = predictions.detach().cpu().numpy()

        # Get real ratings for the user
        user_ratings = user_item_matrix[user_id]

        # Prepare the results list
        results = []

        # Iterate over item IDs and gather information
        for item_id, prediction, real_rating in zip(item_ids, predictions, user_ratings):
            movie_name = movie_names[item_id]  # Get the movie name
            results.append({'movie_name': movie_name, 'predicted_rating': prediction, 'real_rating': real_rating})

        return results
