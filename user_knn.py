import numpy as np

from interface import Regressor
from utils import get_data, Config

from config import *
from scipy.sparse import csc_matrix
from tqdm import tqdm
from collections import defaultdict
from statistics import mode
import pickle
from os import path

class KnnUserSimilarity(Regressor):
    def __init__(self, config):
        self.matrix = None # holds the user-item matrix
        self.corr = defaultdict(list) # holds the correlation of each user with all the other users
        self.k = config.k # hyperparameter k of knn model
        if path.exists(SIMPLE_MEAN_PATH):  # if we have the parameters saved in memory we'll load them
            self.upload_params()

    def fit(self, X: np.array):
        self.matrix = csc_matrix((X[:, RATINGS_COL_INDEX], (X[:, USERS_COL_INDEX],X[:, ITEMS_COL_INDEX]))).toarray() # build the user-item matrix
        if not self.corr: # if the dictionary of the correlation is not exist we'll call functions to build one
            self.build_user_to_user_corr_dict(X)

    def build_user_to_user_corr_dict(self, data):
        for i in tqdm(range(self.matrix.shape[0])): # go over each user
            for j in range(i+1, self.matrix.shape[0]): # go over each user from i+1 in order to save iterations
                user1 = np.array(self.matrix[i, :]) # user 1 vector from user-item matrix
                user2 = np.array(self.matrix[j, :]) # user 2 vector from user-item matrix
                bool_vector = np.array(user1 * user2) > 0 # boolean vector with True only on the indexes where both users ranked
                user1 = user1[bool_vector] # new user 1 vector
                user2 = user2[bool_vector] # new user 2 vector
                if not any(user1 * user2): # if both of the vectors are empty
                    users_corr = 0
                elif np.var(user1) == 0 and np.var(user2) == 0: # both of the vectors are not empty but var is 0
                    users_corr = 1
                elif np.var(user1) == 0 or np.var(user2) == 0: # one of the vectors are not empty but var is 0
                    users_corr = 0
                else:
                    users_corr = np.corrcoef(user1,user2)[0,1] # calculate pearson correlation
                if users_corr >= 0: # if the correlation is non-negative
                    self.corr[i].append((j, users_corr)) # insert value to the dictionary for user 1
                    self.corr[j].append((i, users_corr)) # insert value to the dictionary for user 2
        for user, corr_list in self.corr.items(): # sort each list by its pearson correlation value (high to low)
            self.corr[user] = sorted(corr_list, key = lambda x: x[1], reverse=True)

    def predict_on_pair(self, user: int, item: int):
        if item != -1: # if the item does not exist on the train set
            users_lst = [] # holds the ratings of the k nearest users for the item
            ranked_idxs = np.arange(self.matrix.shape[0])[self.matrix[:,item] > 0] # indexes of users that ranked this item
            for i in self.corr[user]: # go over the correlation list of the user
                if i[0] in ranked_idxs: # check if the user ranked the item
                    users_lst.append(self.matrix[i[0],item]) # if so, we append the rating of the user to 'users_lst'
                if len(users_lst) == self.k: # if we got k nearest users
                    break
            if users_lst: # if there is at least one rating value in the list
                return np.mean(users_lst) # return the mean rating
            return self.user_means[user] # return the mean rating of the user
        else:
            return self.user_means[user] # return the mean rating of the user

    def upload_params(self): # load the user's ratings mean for the cases the item number is -1 or there are no ratings on 'users_lst'
        with open(SIMPLE_MEAN_PATH, 'rb') as file:
            self.user_means = pickle.load(file)

if __name__ == '__main__':
    knn_config = Config(k=10)
    train, validation = get_data()
    train = train[:200000, :]
    knn = KnnUserSimilarity(knn_config)
    knn.fit(train)
    print(knn.calculate_rmse(validation))





