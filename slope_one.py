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
        self.matrix = None
        self.popularity_differences = {}
        if path.exists(POPULARITY_DIFFERENCES_PARAMS_FILE_PATH):
            self.upload_params()

    def fit(self, X: np.array):
        self.matrix = csc_matrix((X[:, RATINGS_COL_INDEX], (X[:, USERS_COL_INDEX],X[:, ITEMS_COL_INDEX]))).toarray()
        if not self.popularity_differences:
            self.build_popularity_difference_dict(X)
            self.save_params()

    def build_popularity_difference_dict(self, data):
        #self.matrix = csc_matrix((data[:, RATINGS_COL_INDEX], (data[:, USERS_COL_INDEX], data[:, ITEMS_COL_INDEX]))).toarray()
        #self.matrix = np.array([[5,3,2],[3,4,0],[0,2,5]])
        for i in tqdm(range(self.matrix.shape[1])):
            for j in range(self.matrix.shape[1]):
                if not self.popularity_differences.get((i,j)):
                    if i != j:
                        item1 = np.array(self.matrix[:,i], dtype=np.int16)
                        item2 = np.array(self.matrix[:,j], dtype=np.int16)
                        values_vector = np.array(item1 - item2, dtype=np.int16)
                        bool_vector = np.array(item1 * item2) > 0
                        bool_vec_sum = np.sum(bool_vector)
                        if bool_vec_sum != 0:
                            pd_vec = np.sum(values_vector * bool_vector, dtype=np.float32) / bool_vec_sum
                        else:
                            pd_vec = 0  # Not a single user that ranked both items # TODO - check this
                        self.popularity_differences[(i,j)] = (pd_vec, np.sum(bool_vector, dtype=np.float16))
                        self.popularity_differences[(j,i)] = (- pd_vec, np.sum(bool_vector, dtype=np.float16))

    def predict_on_pair(self, user: int, item: int):
        # user_items = self.matrix[int(user)]
        user_items = self.matrix[user]
        if item == -1:
            return 3 # TODO - check this
        # item = int(item)
        calc = 0
        total_weight = 0
        for i, rank in enumerate(user_items):
            if rank != 0:
                calc += (user_items[i] + self.popularity_differences[(item, i)][0]) * self.popularity_differences[(item,i)][1]
                total_weight += self.popularity_differences[(item, i)][1]
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