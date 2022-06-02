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
        self.matrix = None
        self.corr = defaultdict(list)
        self.k = config.k
        if path.exists(SIMPLE_MEAN_PATH):
            self.upload_params()

    def fit(self, X: np.array):
        self.matrix = csc_matrix((X[:, RATINGS_COL_INDEX], (X[:, USERS_COL_INDEX],X[:, ITEMS_COL_INDEX]))).toarray()
        if not self.corr:
            self.build_user_to_user_corr_dict(X)

    def build_user_to_user_corr_dict(self, data):
        for i in tqdm(range(self.matrix.shape[0])):
            for j in range(i+1, self.matrix.shape[0]):
                if i != j:
                    user1 = np.array(self.matrix[i, :])
                    user2 = np.array(self.matrix[j, :])
                    bool_vector = np.array(user1 * user2) > 0
                    user1 = user1[bool_vector]
                    user2 = user2[bool_vector]
                    if not any(user1 * user2): # if both of the vectors are empty
                        users_corr = 0
                    elif np.var(user1) == 0 and np.var(user2) == 0: # both of the vectors are not empty but var is 0
                        users_corr = 1
                    elif np.var(user1) == 0 or np.var(user2) == 0: # one of the vectors are not empty but var is 0
                        users_corr = 0
                    else:
                        users_corr = np.corrcoef(user1,user2)[0,1]
                    if users_corr >= 0:
                        self.corr[i].append((j, users_corr))
                        self.corr[j].append((i, users_corr))
        for user, corr_list in self.corr.items():
            self.corr[user] = sorted(corr_list, key = lambda x: x[1], reverse=True)

    def predict_on_pair(self, user: int, item: int):
        if item != -1:
            users_lst = []
            ranked_idxs = np.arange(self.matrix.shape[0])[self.matrix[:,item] > 0]
            for i in self.corr[user]:
                if i[0] in ranked_idxs:
                    users_lst.append(self.matrix[i[0],item])
                if len(users_lst) == self.k:
                    break
            if users_lst:
                return np.mean(users_lst)
            return self.user_means[user]
        else:
            return self.user_means[user]

    def upload_params(self):
        with open(SIMPLE_MEAN_PATH, 'rb') as file:
            self.user_means = pickle.load(file)

if __name__ == '__main__':
    knn_config = Config(k=10)
    train, validation = get_data()
    train = train[:200000, :]
    knn = KnnUserSimilarity(knn_config)
    knn.fit(train)
    print(knn.calculate_rmse(validation))

# import numpy as np
# a = np.array([0,0,0,0])
# c = np.array([3,1,2,3])
# b=a*c>0
# users_corr = np.corrcoef(c,a)
# # b=np.array([True,True,False,True])
# print(a[b])
# print(c[b])
# print(users_corr[0,1])
# import pandas as pd
# df1=pd.DataFrame(a[b])
# df2=pd.DataFrame(c[b])
# print('-----')
# # print(df1)
# # print(df2)
# f=df1.corrwith(df2,method='pearson')
# print(f)

