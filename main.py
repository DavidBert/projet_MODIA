import argparse
import torch
from model import NCF
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path')
    parser.add_argument('--train_path')
    args = parser.parse_args()

    df_train = pd.read_csv(args.train_path + "interactions_train.csv.zip")

    # On garde les 100 recettes qui ont le plus de vote
    df_train["count"] = df_train.groupby("recipe_id").transform('count')['user_id']
    recipeId = df_train.drop_duplicates('recipe_id').sort_values(
        'count', ascending=False).iloc[:100]['recipe_id']
    trainset = df_train[df_train['recipe_id'].isin(recipeId)].reset_index(drop=True)

    # On garde les 100 utilisateurs qui ont le plus voté
    trainset["count"] = trainset.groupby("user_id").transform('count')['recipe_id']
    userId = trainset.drop_duplicates('user_id').sort_values(
        'count', ascending=False).iloc[:20001]['user_id']
    df_train = trainset[trainset['user_id'].isin(userId)].reset_index(drop=True)

    testset = df_train.iloc[-10:]
    trainset = df_train.iloc[:-10]

    n_user = df_train.user_id.nunique()
    n_items = df_train.recipe_id.nunique()

    model = NCF(n_users=n_user, n_items=n_items).to(device) 
    model.load_state_dict(torch.load(args.weights_path, map_location=torch.device(device)))


    testset = testset.reset_index()
    user_list = df_train.user_id.unique() 
    item_list = df_train.recipe_id.unique() 

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

