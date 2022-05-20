import pandas as pd
from config import *
import numpy as np


def get_data():
    """
    reads train, validation to python indices so we don't need to deal with it in each algorithm.
    of course, we 'learn' the indices (a mapping from the old indices to the new ones) only on the train set.
    if in the validation set there is an index that does not appear in the train set then we can put np.nan or
     other indicator that tells us that.
    """
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    #train[USER_COL_NAME_IN_DATAEST] = train[USER_COL_NAME_IN_DATAEST].apply(lambda x: x-1)
    #validation[USER_COL_NAME_IN_DATAEST].apply(lambda x:x-1)
    train[USER_COL_NAME_IN_DATAEST] = train[USER_COL_NAME_IN_DATAEST] - 1  #changing the user ids from 0 to n-1 by decreasing all by 1 - Train
    validation[USER_COL_NAME_IN_DATAEST] = validation[USER_COL_NAME_IN_DATAEST] - 1  #changing the user ids from 0 to n-1 by decreasing all by 1 - Validation
    uniqe_lst = list(train[ITEM_COL_NAME_IN_DATASET].unique()) #list of unique items from train data
    new_ids = validation[validation[ITEM_COL_NAME_IN_DATASET].isin(uniqe_lst) == False]  #holding the rows in validation data that have an item that doesnt exists in train data
    index_to_change = new_ids.index  #holding the index of these rows
    validation[ITEM_COL_NAME_IN_DATASET][index_to_change] = np.nan
    return train, validation


class Config:
    def __init__(self, **kwargs):
        self._set_attributes(kwargs)

    def _set_attributes(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
print(get_data())