import torch.nn as nn
import torch
from torch.utils.data import Dataset

class Ratings_Datset(Dataset):
    def __init__(self, df, user2id, item2id):
        self.df = df.reset_index()
        self.user2id = user2id
        self.item2id = item2id

    def __len__(self):
        return len(self.df)
  
    def __getitem__(self, idx):
        user = self.user2id[self.df['user_id'][idx]]
        user = torch.tensor(user, dtype=torch.long)
        item = self.item2id[self.df['recipe_id'][idx]]
        item = torch.tensor(item, dtype=torch.long)
        rating = torch.tensor(self.df['rating'][idx], dtype=torch.float)
        return user, item, rating

class NCF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=8):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        self.predictor = nn.Sequential(
            nn.Linear(in_features=n_factors * 2, out_features=64),
            nn.Linear(in_features=64, out_features=32),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, user, item):

        u = self.user_embeddings(user)
        i = self.item_embeddings(item)

        # Concat the two embedding layers
        z = torch.cat([u, i], dim=-1)
        return self.predictor(z)
