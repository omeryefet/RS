from config import TRAIN_PATH,VALIDATION_PATH,USER_COL,ITEM_COL,RATING_COL
import pandas as pd
import numpy as np


'''This func returns train and validation data sets with reindexed user_ids and item_ids'''
def get_data():
    train_data= pd.read_csv(TRAIN_PATH)
    val_data = pd.read_csv(VALIDATION_PATH)
    train_data["User_ID_Alias"] = train_data["User_ID_Alias"] - 1                    #changing the user ids from 0 to n-1 by decreasing all by 1 - Train
    val_data["User_ID_Alias"] = val_data["User_ID_Alias"] - 1                        #changing the user ids from 0 to n-1 by decreasing all by 1 - Validation
    uniqe_lst = list(train_data["Movie_ID_Alias"].unique())                          #list of unique items from train data
    new_ids = val_data[val_data["Movie_ID_Alias"].isin(uniqe_lst) == False]          #holding the rows in validation data that have an item that doesnt exists in train data
    index_to_change = new_ids.index                                                  #holding the index of these rows
    val_data["Movie_ID_Alias"][index_to_change] = np.nan                              #replacing these item indexes with None

    '''Check if there are users in validation and not in train- the results shows that there is no states like this'''
    # lst = list(val_data["User_ID_Alias"].unique())
    # user_in_val_not_train = train_data[train_data["User_ID_Alias"].isin(lst) == False]
    # print(user_in_val_not_train)

    item_unique = sorted(uniqe_lst)                                                  #list of sorted unique items from train data
    new_items_index = list(range(len(item_unique)))                                  #list of new item ids (in ascending and continuous order)
    '''we replaced the same items with the same new item indexes in both train and validation:'''
    train_data["Movie_ID_Alias"].replace(item_unique, new_items_index, inplace=True) #replacing each unique item with a new item id (in ascending and continuous order) - Train
    val_data["Movie_ID_Alias"].replace(item_unique, new_items_index, inplace=True)   #replacing each unique item with a new item id (in ascending and continuous order) - Validation
    train_data.rename(columns={"User_ID_Alias": USER_COL,"Movie_ID_Alias": ITEM_COL, 'Ratings_Rating':RATING_COL}, inplace=True)
    val_data.rename(columns={"User_ID_Alias": USER_COL, "Movie_ID_Alias": ITEM_COL, 'Ratings_Rating':RATING_COL}, inplace=True)
    val_data= val_data.to_numpy()

    return train_data, val_data


class Config:
    def __init__(self, **kwargs):
        self._set_attributes(kwargs)

    def _set_attributes(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)