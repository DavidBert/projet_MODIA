
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model import NCF
import pickle


class Ratings_Dataset(Dataset):
    def __init__(self, df, user2id, item2id, n_factors):
        self.df = df.reset_index()
        self.user2id = user2id
        self.item2id = item2id
        self.n_factors = n_factors

    def __len__(self):
        return len(self.df)
  
    def __getitem__(self, idx):
        user = self.user2id[self.df['user_id'][idx]]
        user = torch.tensor(user, dtype=torch.long)
        item = self.item2id[self.df['recipe_id'][idx]]
        item = torch.tensor(item, dtype=torch.long)
        return user, item


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, help="weights path")
    parser.add_argument('--test_path', type=str, help="test file path")

    args = parser.parse_args()
    weights_path = args.weights_path
    test_path = args.test_path

    dict_weights = torch.load(weights_path)
    n_users = dict_weights["user_embeddings.weight"].shape[0]
    n_items = dict_weights["item_embeddings.weight"].shape[0]
    n_factors = dict_weights["user_embeddings.weight"].shape[1]
    model = NCF(n_users, n_items, n_factors)

    testset = pd.read_csv(test_path)
    user2id, item2id = pickle.load(open('mapping.pkl', 'rb'))

    testloader = DataLoader(Ratings_Dataset(testset, user2id, item2id, n_factors), batch_size=64, num_workers=2)
    users, recipes = next(iter(testloader))
    users = users.to(device)
    recipes = recipes.to(device)

    y = model(users, recipes)*5

    id2user = {v: k for k, v in user2id.items()}
    id2item = {v: k for k, v in item2id.items()}
    print("\nRating predictions\n")
    for user, recipe, y_ in zip(users, recipes, y):
        print(f'User {id2user[user.data.item()]:7d} and recipe {id2item[recipe.data.item()]:7d} -> r = {y_.data.item():.2f}')
