import argparse
import torch
from model import NCF
import pandas as pd
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path')
    parser.add_argument('--test_path')
    args = parser.parse_args()

    testset = pd.read_csv(args.test_path)

    model = NCF(n_users=n_user, n_items=n_items).to(device) 
    model.load_state_dict(torch.load(args.weights_path, map_location=torch.device(device)))

    testset = testset.reset_index()

    user2id = {w: i for i, w in enumerate(user_list)}
    item2id = {w: i for i, w in enumerate(item_list)}

    for idx in range(testset.shape[0]): 
        user = user2id[testset['user_id'][idx]]
        user = torch.tensor(user, dtype=torch.long)
        item = item2id[testset['recipe_id'][idx]]
        item = torch.tensor(item, dtype=torch.long)
        rating = torch.tensor(testset['rating'][idx], dtype=torch.float)

        y = model(user, item)
        print(f"The user with id {testset['user_id'][idx]} predicts the rate of {y} instead of {rating} for movie id {testset['recipe_id'][idx]}")
