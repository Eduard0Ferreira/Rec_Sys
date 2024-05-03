import torch
from torch import nn, optim
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MF(nn.Module):

    def __init__(self, num_users, num_items, emb_dim, init):
        super().__init__()
        self.user_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=emb_dim)
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=emb_dim)

        self.fc1 = nn.Linear(emb_dim, emb_dim * 4)
        self.fc2 = nn.Linear(emb_dim * 4, emb_dim * 2)
        self.fc3 = nn.Linear(emb_dim * 2, 1)
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

        # Reshape element_product to match the input size of self.fc1
        element_product = element_product.unsqueeze(1)  # Add an extra dimension
        # Expand along the second dimension to match fc1 input size
        element_product = element_product.expand(-1, self.fc1.in_features)

        x = self.relu(self.fc1(element_product))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        user_b = self.user_bias[user]
        item_b = self.item_bias[item]
        # element_product += user_b + item_b + self.offset

        return x.squeeze() + user_b + item_b + self.offset

    def fit(self, num_epochs, train_loader, test_loader):

        # train the model
        loss_fun = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=.001)
        # initialize accuracies as empties
        train_losses = []
        val_losses = []

        # loop over epochs
        for epoch in range(num_epochs):

            # activate training mode
            self.train()

            # loop over training data batches
            batch_loss = []
            for X, y in train_loader:
                # forward pass and loss
                user, item = X[:, 0], X[:, 1]
                y_rating = y.to(device, dtype=torch.float)
                y_hat = self(user, item)
                loss = loss_fun(y_hat, y_rating)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            # append the batch loss
            train_losses.append(np.mean(loss.item()))
            print(f'epoch {epoch + 1} loss batch: {np.mean(batch_loss)}')
            # activate testing mode
            self.eval()

            x, y = next(iter(test_loader))
            user_val, item_val = x[:, 0], x[:, 1]
            with torch.no_grad():
                y_hat = self(user_val, item_val)
                loss = loss_fun(y_hat, y)
            val_losses.append(loss)

        return train_losses, val_losses
