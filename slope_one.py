from interface import Regressor
from utils import get_data
import numpy as np
from config import *
from scipy.sparse import csc_matrix
import pickle
from os import path
from tqdm import tqdm

class SlopeOne(Regressor):
    def __init__(self):
        self.matrix = None # holds the user-item matrix
        self.popularity_differences = {} # holds the popularity difference of each pair of items
        if path.exists(POPULARITY_DIFFERENCES_PARAMS_FILE_PATH): # if we have the parameters saved in memory we'll load them
            self.upload_params()

    def fit(self, X: np.array):
        self.matrix = csc_matrix((X[:, RATINGS_COL_INDEX], (X[:, USERS_COL_INDEX],X[:, ITEMS_COL_INDEX]))).toarray() # build the user-item matrix
        if not self.popularity_differences: # if the dictionary of the popularity difference is not exist we'll call functions to build one and save it
            self.build_popularity_difference_dict(X)
            self.save_params()

    def build_popularity_difference_dict(self, data):
        for i in tqdm(range(self.matrix.shape[1])): # go over each pair of items (2 for loops)
            for j in range(self.matrix.shape[1]):
                if not self.popularity_differences.get((i,j)): # if the key is not exist yet
                    if i != j: # only when we have 2 different users
                        item1 = np.array(self.matrix[:,i]) # item 1 vector from user-item matrix
                        item2 = np.array(self.matrix[:,j]) # item 2 vector from user-item matrix
                        values_vector = np.array(item1 - item2) # vector of item 1 - item 2
                        bool_vector = np.array(item1 * item2) > 0 # boolean vector with True only on the indexes where both items was ranked
                        bool_vec_sum = np.sum(bool_vector) # sum of the boolean vector
                        if bool_vec_sum != 0: # if there is at least one user that ranked both items
                            pd_vec = np.sum(values_vector * bool_vector) / bool_vec_sum # calculate the mean
                        else: # Not a single user that ranked both items
                            pd_vec = 0
                        self.popularity_differences[(i,j)] = (pd_vec, bool_vec_sum) # insert values to the dictionary - (popularity differece value, the number of users we calculate the mean of)
                        self.popularity_differences[(j,i)] = (- pd_vec, bool_vec_sum)

    def predict_on_pair(self, user: int, item: int):
        user_items = self.matrix[user]
        if item == -1: # if the item does not exist on the train set
            return np.mean(user_items) # return the mean ranking of the user
        calc = 0
        total_weight = 0
        for i, rank in enumerate(user_items): # go over the items the user ranked
            if rank != 0:
                calc += (rank + self.popularity_differences[(item, i)][0]) * self.popularity_differences[(item,i)][1] # calculate the numerator of the predicted ranking
                total_weight += self.popularity_differences[(item, i)][1] # calculate the denominator of the predicted ranking
        if (calc/total_weight) > 5:
            return 5
        elif (calc/total_weight) < 0:
            return 0
        else:
            return calc/total_weight

    def upload_params(self):
        with open(POPULARITY_DIFFERENCES_PARAMS_FILE_PATH, 'rb') as file:
            self.popularity_differences = pickle.load(file)

    def save_params(self):
        with open(POPULARITY_DIFFERENCES_PARAMS_FILE_PATH, 'wb') as file:
            pickle.dump(self.popularity_differences, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    train, validation = get_data()
    slope_one = SlopeOne()
    slope_one.fit(train)
    print(slope_one.calculate_rmse(validation))