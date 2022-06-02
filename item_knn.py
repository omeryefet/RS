import numpy as np

from interface import Regressor
from utils import get_data, Config

import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from config import *
from os import path
import pickle
import csv
from scipy.sparse import coo_matrix
from tqdm import tqdm
import math


class KnnItemSimilarity(Regressor):
    def __init__(self, config):
        # self.corr_matrix = None  # The similarity matrix for all the pairs of items
        self.k = config.k
        self.n_users = None
        self.n_items = None
        # self.item_avg_rank = None  # The average rating for each item
        # self.sim_matrix = None  # Similarity matrix but with sparse matrix
        # self.rank_matrix = None  # Matrix with users in rows, items in columns and rating as the value
        # self.rank_matrix_spars = None  # The same matrix as before but with sparse matrix
        # self.user_avg_rank = None  # # The average rating for each user
        if path.exists(CORRELATION_PARAMS_FILE_PATH):
            self.upload_params()

    def fit(self, X: np.array):
        self.n_users = X[USER_COL].nunique()
        self.n_items = X[ITEM_COL].nunique()
        self.corr_matrix = np.zeros((self.n_items, self.n_items))
        self.user_avg_rank = X.groupby(USER_COL)[RATING_COL].mean()                     # The average rating for each user
        self.item_avg_rank = X.groupby(ITEM_COL)[RATING_COL].mean()                     # The average rating for each item
        self.rank_matrix = train.pivot(index=USER_COL, columns=ITEM_COL, values=RATING_COL)  # Create matrix with users in rows, items in columns and rating as the value
        self.rank_matrix_spars = csr_matrix(self.rank_matrix)                          # Save the matrix in sparse matrix
        if path.exists(CORRELATION_PARAMS_FILE_PATH):
            self.upload_params()
            spars_df = self.sim_matrix.toarray()  # Transform the similarity sparse matrix to numpy_array
            for i in range(spars_df.shape[0]):
                item1 = spars_df[i, 0]
                item2 = spars_df[i, 1]
                self.corr_matrix[int(item1), int(item2)] = spars_df[i, 2]  # Save the similarity in corr_matrix from the csv file- include items with no similarity
            self.corr_matrix = self.corr_matrix + self.corr_matrix.T  # Similarity for (x,y) is the same as (y,x)
            self.corr_matrix = np.nan_to_num(self.corr_matrix)  # Change nan to 0
        else:
            self.build_item_to_itm_corr_dict()
            self.save_params()

        # if not self.corr:
        #     self.build_user_to_user_corr_dict(X)
        #     self.save_params()

    def build_item_to_itm_corr_dict(self, data):
        raise NotImplementedError


    def predict_on_pair(self, user, item):
        raise NotImplementedError


    def upload_params(self):
        with open(CORRELATION_PARAMS_FILE_PATH, 'rb') as file:
            self.corr = pickle.load(file)

    def save_params(self):
        with open(CORRELATION_PARAMS_FILE_PATH, 'wb') as file:
            pickle.dump(self.corr, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    knn_config = Config(k=25)
    train, validation = get_data()
    knn = KnnItemSimilarity(knn_config)
    knn.fit(train)
    print(knn.calculate_rmse(validation))
