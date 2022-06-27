from model import NCF, Ratings_Datset
import sys
import pandas as pd 
import torch
from torch.utils.data import DataLoader


def main(test_path):
    test_path = test_path[1] 
    
    dict_weight = torch.load('weight.pth', map_location=torch.device('cpu'))
    
    to_predict = pd.read_csv(test_path)
    
    n_user = dict_weight['user_embeddings.weight'].size()[0]
    n_items = dict_weight['item_embeddings.weight'].size()[0]
    
    model = NCF(n_user, n_items)
    model.load_state_dict(dict_weight)

    user_list = to_predict.user_id.unique()
    item_list = to_predict.recipe_id.unique()
    user2id = {w: i for i, w in enumerate(user_list)}
    item2id = {w: i for i, w in enumerate(item_list)}    
    
    testloader = DataLoader(Ratings_Datset(to_predict, user2id, item2id), batch_size=1)     
    
    for user, recipe, r in iter(testloader):
        y = model(user, recipe)*5
        print("utilisateur n°", user_list[user.numpy()[0]], " donne la note ",y.round(decimals=2).detach().numpy()[0][0], " à la recette n°", item_list[recipe.numpy()[0]])
        

if __name__ == "__main__":
    main(sys.argv)
    