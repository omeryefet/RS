import numpy as np
from config import BASELINE_PARAMS_FILE_PATH,CORRELATION_PARAMS_FILE_PATH
from interface import Regressor
from utils import get_data, Config
import pandas as pd
from scipy.sparse import csr_matrix
import math


class KnnBaseline(Regressor):
    def __init__(self, config):
        self.bu = None
        self.bi = None
        self.rank_matrix = None            # Matrix with users in rows, items in columns and rating as the value
        self.global_bias = None
        self.k = config.k
        self.corr_matrix = None            # The similarity matrix for all the pairs of items
        self.item_avg_rank = None          # The average rating for each item
        self.n_users = None
        self.n_items = None
        self.user_avg_rank = None

    def fit(self, X: np.array):
        self.n_users = X["user"].nunique()
        self.n_items = X["item"].nunique()
        self.corr_matrix = np.zeros((self.n_items, self.n_items))
        self.upload_params()
        self.preproccessing(X)

    def preproccessing(self, data):
        self.user_avg_rank = data.groupby('user')['rating'].mean()                  # The average rating for each user
        self.item_avg_rank = data.groupby('item')['rating'].mean()                  # The average rating for each item
        train_to_pivot = data.pivot(index='user', columns='item', values='rating')  # Create matrix with users in rows, items in columns and rating as the value
        train_to_pivot.fillna(0, inplace=True)
        train_to_pivot = train_to_pivot.astype('int32')
        self.rank_matrix = csr_matrix(train_to_pivot)                               # Save the matrix in sparse matrix

    def predict_on_pair(self, user: int, item: int):
        user = user.astype('int')
        if math.isnan(item):
            return self.user_avg_rank[user]                         # If the item is nan predict by the average of the user
        item = item.astype('int')
        b_ui = self.global_bias + self.bu[user] + self.bi[item]
        sims = self.corr_matrix[item, :]                            # Array of the similarity for each item with the item we want to predict
        common_items = np.argsort(-sims)[:self.k]                   # Get the top K's index higher similarity
        sum_numerator = 0
        sum_denominator = 0
        for i in common_items:
            b_uj = self.global_bias + self.bu[user] + self.bi[i]    # Take the similarity between the 2 items
            if self.corr_matrix[item, i] != 0:                      # If the rating for the chosen item is not 0
                common_sim = self.corr_matrix[item, i]              # Take the similarity between the 2 items
                rating = self.rank_matrix[user, i]                  # Take the rating for the user with item i- the item from the common_items
                sum_numerator += (common_sim * (rating-b_uj[0]))
                sum_denominator += common_sim
        if (sum_denominator > 0) & (sum_numerator > 0):             # If we can divide for predict- there are at least one item with positive similarity
            predict = b_ui[0] + (sum_numerator / sum_denominator)
            return predict
        else:                                                       # If we can't divide so predict the average rating for the item
            return self.item_avg_rank[item]

    def upload_params(self):
        data_bias = pd.read_pickle(BASELINE_PARAMS_FILE_PATH)
        self.bu = data_bias[0]
        self.bi = data_bias[1]
        self.global_bias = data_bias[2]
        data_sim = pd.read_csv(CORRELATION_PARAMS_FILE_PATH)
        spars_df = csr_matrix(data_sim).toarray()                      # Transform the similarity sparse matrix to numpy_array
        for i in range(spars_df.shape[0]):
            item1 = spars_df[i, 0]
            item2 = spars_df[i, 1]
            self.corr_matrix[int(item1), int(item2)] = spars_df[i, 2]  # Save the similarity in corr_matrix from the csv file- include items with no similarity
        self.corr_matrix = self.corr_matrix + self.corr_matrix.T       # Similarity for (x,y) is the same as (y,x)
        self.corr_matrix = np.nan_to_num(self.corr_matrix)             # Change nan to 0


if __name__ == '__main__':
    baseline_knn_config = Config(k=25)
    train, validation = get_data()
    knn_baseline = KnnBaseline(baseline_knn_config)
    knn_baseline.fit(train)
    print(knn_baseline.calculate_rmse(validation))
