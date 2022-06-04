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

    unique_list_user_id_train_after = list(range(train[USER_COL_NAME_IN_DATAEST].nunique()))
    unique_list_item_id_train_after = list(range(train[ITEM_COL_NAME_IN_DATASET].nunique()))
    unique_list_user_id_train_before = sorted(list(train[USER_COL_NAME_IN_DATAEST].unique()))
    unique_list_item_id_train_before = sorted(list(train[ITEM_COL_NAME_IN_DATASET].unique()))

    index_to_change_item = validation[validation[ITEM_COL_NAME_IN_DATASET].isin(list(train[ITEM_COL_NAME_IN_DATASET].unique())) == False].index
    validation[ITEM_COL_NAME_IN_DATASET][index_to_change_item] = -1
    index_to_change_user = validation[validation[USER_COL_NAME_IN_DATAEST].isin(list(train[USER_COL_NAME_IN_DATAEST].unique())) == False].index
    validation[USER_COL_NAME_IN_DATAEST][index_to_change_user] = -1

    train[USER_COL_NAME_IN_DATAEST].replace(unique_list_user_id_train_before , unique_list_user_id_train_after , inplace=True)
    validation[USER_COL_NAME_IN_DATAEST].replace(unique_list_user_id_train_before , unique_list_user_id_train_after , inplace=True)
    train[ITEM_COL_NAME_IN_DATASET].replace(unique_list_item_id_train_before , unique_list_item_id_train_after , inplace=True)
    validation[ITEM_COL_NAME_IN_DATASET].replace(unique_list_item_id_train_before , unique_list_item_id_train_after , inplace=True)

    train.rename(columns={USER_COL_NAME_IN_DATAEST: USER_COL,ITEM_COL_NAME_IN_DATASET: ITEM_COL, RATING_COL_NAME_IN_DATASET:RATING_COL}, inplace=True)

    train = train.to_numpy(dtype=int)
    validation = validation.to_numpy(dtype=int)
    return train, validation


class Config:
    def __init__(self, **kwargs):
        self._set_attributes(kwargs)

    def _set_attributes(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)