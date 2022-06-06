import numpy as np

from interface import Regressor
from utils import get_data, Config

import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from config import *
from os import path
import pickle
from collections import defaultdict
import csv
from scipy.sparse import coo_matrix
from tqdm import tqdm
import math


class KnnItemSimilarity(Regressor):
    def __init__(self, config):
        self.matrix = None # holds the user-item matrix
        self.corr = defaultdict(list) # holds the correlation of each item with all the other items
        self.k = config.k # hyperparameter k of knn model
        if path.exists(CORRELATION_PARAMS_FILE_PATH) and path.exists(SIMPLE_MEAN_PATH): # if we have the parameters saved in memory we'll load them
            self.upload_params()

    def fit(self, X: np.array):
        self.matrix = csc_matrix((X[:, RATINGS_COL_INDEX], (X[:, USERS_COL_INDEX],X[:, ITEMS_COL_INDEX]))).toarray() # build the user-item matrix
        if not self.corr: # if the dictionary of the correlation is not exist we'll call functions to build one and save it
            self.build_item_to_item_corr_dict(X)
            self.save_params()

    def build_item_to_item_corr_dict(self, data):
        for i in tqdm(range(self.matrix.shape[1])): # go over each item
            for j in range(i+1, self.matrix.shape[1]): # go over each item from i+1 in order to save iterations
                item1 = np.array(self.matrix[:, i]) # item 1 vector from user-item matrix
                item2 = np.array(self.matrix[:, j]) # item 2 vector from user-item matrix
                bool_vector = np.array(item1 * item2) > 0 # boolean vector with True only on the indexes where both items was ranked
                item1 = item1[bool_vector] # new item 1 vector
                item2 = item2[bool_vector] # new item 2 vector
                if not any(item1 * item2): # if both of the vectors are empty
                    items_corr = 0
                elif np.var(item1) == 0 and np.var(item2) == 0: # both of the vectors are not empty but var is 0
                    items_corr = 1
                elif np.var(item1) == 0 or np.var(item2) == 0: # one of the vectors are not empty but var is 0
                    items_corr = 0
                else:
                    items_corr = np.corrcoef(item1,item2)[0,1] # calculate pearson correlation
                if items_corr >= 0: # if the correlation is non-negative
                    self.corr[i].append((j, items_corr)) # insert value to the dictionary for item 1
                    self.corr[j].append((i, items_corr)) # insert value to the dictionary for item 2
        for item, corr_list in self.corr.items(): # sort each list by its pearson correlation value (high to low)
            self.corr[item] = sorted(corr_list, key = lambda x: x[1], reverse=True)


    def predict_on_pair(self, user, item):
        if item != -1: # if the item does not exist on the train set
            items_lst = [] # holds the ratings of the k nearest items for the user
            ranked_idxs = np.arange(self.matrix.shape[1])[self.matrix[user,:] > 0] # indexes of items that ranked by this user
            for i in self.corr[item]: # go over the correlation list of the item
                if i[0] in ranked_idxs: # check if the item was ranked by the user
                    items_lst.append(self.matrix[user,i[0]]) # if so, we append the rating of the user to 'items_lst'
                if len(items_lst) == self.k: # if we got k nearest items
                    break
            if items_lst: # if there is at least one rating value in the list
                return np.mean(items_lst) # return the mean rating
            return self.user_means[user] # return the mean rating of the user
        else:
            return self.user_means[user] # return the mean rating of the user

    def upload_params(self):
        with open(CORRELATION_PARAMS_FILE_PATH, 'rb') as file:
            self.corr = pickle.load(file)
        with open(SIMPLE_MEAN_PATH, 'rb') as file: # load the user's ratings mean for the cases the item number is -1 or there are no ratings on 'users_lst'
            self.user_means = pickle.load(file)

    def save_params(self):
        with open(CORRELATION_PARAMS_FILE_PATH, 'wb') as file:
            pickle.dump(self.corr, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    knn_config = Config(k=25)
    train, validation = get_data()
    knn = KnnItemSimilarity(knn_config)
    knn.fit(train)
    print(knn.calculate_rmse(validation))
