import argparse
import torch
from model import NCF
import pandas as pd
import numpy as np


#PATH_TRAIN = '/content/interactions_train.csv.zip'
n_user = 12725
n_items = 100

item_list = np.load('/content/item_list.npy')
user_list = np.load('/content/user_list.npy')


#df_interactions_train = pd.read_csv(PATH_TRAIN)
# On garde les 100 recettes qui ont le plus de vote
#df_interactions_train["count"] = df_interactions_train.groupby("recipe_id").transform('count')['user_id']
#recipeId = df_interactions_train.drop_duplicates('recipe_id').sort_values(
#    'count', ascending=False).iloc[:100]['recipe_id']
#df_interactions_train = df_interactions_train[df_interactions_train['recipe_id'].isin(recipeId)].reset_index(drop=True)

# On garde les 100 utilisateurs qui ont le plus voté
#df_interactions_train["count"] = df_interactions_train.groupby("user_id").transform('count')['recipe_id']
#userId = df_interactions_train.drop_duplicates('user_id').sort_values(
#    'count', ascending=False).iloc[:20001]['user_id']
#df_interactions_train = df_interactions_train[df_interactions_train['user_id'].isin(userId)].reset_index(drop=True)

#user_list = df_interactions_train.user_id.unique()
#item_list = df_interactions_train.recipe_id.unique()
