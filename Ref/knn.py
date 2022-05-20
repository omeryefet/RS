import numpy as np
import pandas as pd
from interface import Regressor
from utils import get_data, Config
from scipy.sparse import csr_matrix
from config import CORRELATION_PARAMS_FILE_PATH,CSV_COLUMN_NAMES
from os import path
import csv
from scipy.sparse import coo_matrix
from tqdm import tqdm
import math


class KnnItemSimilarity(Regressor):
    def __init__(self, config):
        self.corr_matrix = None             # The similarity matrix for all the pairs of items
        self.k = config.k
        self.n_users = None
        self.n_items = None
        self.item_avg_rank = None           # The average rating for each item
        self.sim_matrix = None              # Similarity matrix but with sparse matrix
        self.rank_matrix = None             # Matrix with users in rows, items in columns and rating as the value
        self.rank_matrix_spars = None       # The same matrix as before but with sparse matrix
        self.user_avg_rank = None           # # The average rating for each user

    def fit(self, X:np.array):
        self.n_users = X["user"].nunique()
        self.n_items = X["item"].nunique()
        self.corr_matrix = np.zeros((self.n_items, self.n_items))
        self.preproccessing(X)
        if path.exists(CORRELATION_PARAMS_FILE_PATH):
            self.upload_params()
            spars_df = self.sim_matrix.toarray()                                           # Transform the similarity sparse matrix to numpy_array
            for i in range(spars_df.shape[0]):
                item1 = spars_df[i, 0]
                item2 = spars_df[i, 1]
                self.corr_matrix[int(item1), int(item2)] = spars_df[i, 2]                  # Save the similarity in corr_matrix from the csv file- include items with no similarity
            self.corr_matrix = self.corr_matrix + self.corr_matrix.T                       # Similarity for (x,y) is the same as (y,x)
            self.corr_matrix = np.nan_to_num(self.corr_matrix)                             # Change nan to 0
        else:
            self.build_item_to_itm_corr_dict()
            self.save_params()

    def build_item_to_itm_corr_dict(self):
        self.rank_matrix = self.rank_matrix.reset_index().rename_axis(None, axis=1)                         # Convert the rank matrix into Data Frame
        self.rank_matrix = self.rank_matrix.set_index('user')                                               # Use users as indexes
        for i in tqdm(range(self.n_items)):                                                                 # For each item
            x = np.sum(self.rank_matrix.multiply(self.rank_matrix[i], axis='rows').notnull(), axis=0) > 1   # Get true if the item rated by above 1 user and False-else
            all_indexes = x[x].index                                                                        # Saves all items rated by users with the current item above 1 (True rows)
            good_indexes = all_indexes[all_indexes>i]                                                       # Save only the indexes that above item i because the rest we computed
            i_corr = self.rank_matrix.loc[:, good_indexes].corrwith(self.rank_matrix[i], method='pearson')  # Compute similarity for the current item with all the items that come after and have rated by above 1 user - because for the rest we computed
            i_corr = i_corr[i_corr.notnull()]
            indexes = list(i_corr.index)                                                      # Get all the indexes with not nan in their similarity
            self.corr_matrix[i, indexes] = i_corr                                             # Fill corr matrix with all the similarities we found with the current item
            self.corr_matrix[i, i] = 0                                                        # Do not use similarity=1 for the item with itself in the common items for the predict
        self.corr_matrix = np.nan_to_num(self.corr_matrix)                                    # Change nan to zeros
        self.corr_matrix = self.corr_matrix.clip(min=0)                                       # Save similarity>0
        self.sim_matrix = csr_matrix(self.corr_matrix)                                        # Save diagonal matrix for the csv file
        self.corr_matrix = self.corr_matrix + self.corr_matrix.T                              # Similarity for (x,y) is the same as (y,x)

    def preproccessing(self, data):
        self.user_avg_rank = data.groupby('user')['rating'].mean()                     # The average rating for each user
        self.item_avg_rank = data.groupby('item')['rating'].mean()                     # The average rating for each item
        self.rank_matrix = train.pivot(index='user', columns='item', values='rating')  # Create matrix with users in rows, items in columns and rating as the value
        self.rank_matrix_spars = csr_matrix(self.rank_matrix)                          # Save the matrix in sparse matrix

    def predict_on_pair(self, user, item):
        user = user.astype('int')
        if math.isnan(item):
            return self.user_avg_rank[user]                   # If the item is nan predict by the average of the user
        item = item.astype('int')
        sims = self.corr_matrix[item, :]                      # Array of the similarity for each item with the item we want to predict
        common_items = np.argsort(-sims)[:self.k]             # Get the top K's index higher similarity
        sum_numerator = 0
        sum_denominator = 0
        for i in common_items:
            if self.corr_matrix[item, i] != 0:                # If the rating for the chosen item is not 0
                common_sim = self.corr_matrix[item, i]        # Take the similarity between the 2 items
                rating = self.rank_matrix_spars[user, i]      # Take the rating for the user with item i- the item from the common_items
                sum_numerator += (common_sim * rating)
                sum_denominator += common_sim
        if (sum_denominator > 0) & (sum_numerator > 0):       # If we can divide for predict- there are at least one item with positive similarity
            predict = sum_numerator / sum_denominator
            return predict
        else:
            return self.item_avg_rank[item]                   # If we can't divide so predict the average rating for the item

    def upload_params(self):
        data_read=pd.read_csv(CORRELATION_PARAMS_FILE_PATH)
        self.sim_matrix = csr_matrix(data_read)               # Save the similarity as sparse matrix

    def save_params(self):
        with open(CORRELATION_PARAMS_FILE_PATH, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(CSV_COLUMN_NAMES)
            A = coo_matrix(self.sim_matrix)
            for i, j, v in zip(A.row, A.col, A.data):  # Save the items and similarity from the sparse matrix separately
                i = i.astype('int16')
                j = j.astype('int16')
                v = v.astype('float32')
                writer.writerow([i, j, round(v, 4)])   # Write the 3 elements to csv file and round the similarity


if __name__ == '__main__':
    knn_config = Config(k=25)
    train, validation = get_data()
    knn = KnnItemSimilarity(knn_config)
    knn.fit(train)
    print(knn.calculate_rmse(validation))
