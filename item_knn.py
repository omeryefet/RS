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
        self.matrix = None
        self.corr = defaultdict(list)
        self.k = config.k
        if path.exists(CORRELATION_PARAMS_FILE_PATH) and path.exists(SIMPLE_MEAN_PATH):
            self.upload_params()

    def fit(self, X: np.array):
        self.matrix = csc_matrix((X[:, RATINGS_COL_INDEX], (X[:, USERS_COL_INDEX],X[:, ITEMS_COL_INDEX]))).toarray()
        if not self.corr:
            self.build_item_to_item_corr_dict(X)
            self.save_params()

    def build_item_to_item_corr_dict(self, data):
        for i in tqdm(range(self.matrix.shape[1])):
            for j in range(i+1, self.matrix.shape[1]):
                if i != j:
                    item1 = np.array(self.matrix[:, i])
                    item2 = np.array(self.matrix[:, j])
                    bool_vector = np.array(item1 * item2) > 0
                    item1 = item1[bool_vector]
                    item2 = item2[bool_vector]
                    if not any(item1 * item2): # if both of the vectors are empty
                        items_corr = 0
                    elif np.var(item1) == 0 and np.var(item2) == 0: # both of the vectors are not empty but var is 0
                        items_corr = 1
                    elif np.var(item1) == 0 or np.var(item2) == 0: # one of the vectors are not empty but var is 0
                        items_corr = 0
                    else:
                        items_corr = np.corrcoef(item1,item2)[0,1]
                    if items_corr >= 0:
                        self.corr[i].append((j, items_corr))
                        self.corr[j].append((i, items_corr))
        for item, corr_list in self.corr.items():
            self.corr[item] = sorted(corr_list, key = lambda x: x[1], reverse=True)


    def predict_on_pair(self, user, item):
        if item != -1:
            items_lst = []
            ranked_idxs = np.arange(self.matrix.shape[1])[self.matrix[user,:] > 0]
            for i in self.corr[item]:
                if i[0] in ranked_idxs:
                    items_lst.append(self.matrix[user,i[0]])
                if len(items_lst) == self.k:
                    break
            if items_lst:
                return np.mean(items_lst)
            return self.user_means[user]
        else:
            return self.user_means[user]

    def upload_params(self):
        with open(CORRELATION_PARAMS_FILE_PATH, 'rb') as file:
            self.corr = pickle.load(file)
        with open(SIMPLE_MEAN_PATH, 'rb') as file:
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
