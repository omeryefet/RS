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
        if path.exists(CORRELATION_PARAMS_FILE_PATH):
            self.upload_params()

    def fit(self, X: np.array):
        X = X.to_numpy(dtype=int)
        self.matrix = csc_matrix((X[:, RATINGS_COL_INDEX], (X[:, USERS_COL_INDEX],X[:, ITEMS_COL_INDEX]))).toarray()
        if not self.corr:
            self.build_item_to_itm_corr_dict(X)
            self.save_params()


    def build_item_to_itm_corr_dict(self, data):
        pairs = set()
        for i in tqdm(range(self.matrix.shape[0])):
            for j in range(i+1, self.matrix.shape[0]):
                pairs.add((i,j))
                if i != j:
                    user1 = np.array(self.matrix[i, :], dtype=np.int16)
                    user2 = np.array(self.matrix[j, :], dtype=np.int16)
                    bool_vector = np.array(user1 * user2) > 0
                    user1 = user1[bool_vector]
                    user2 = user2[bool_vector]
                    if any(user1 * user2):
                        # users_corr = np.corrcoef(user1,user2)[0,1]
                        users_corr = 1
                    else:
                        users_corr = 0
                    self.corr[i].append((j,users_corr))
                    # self.corr[j].append((i,users_corr))
        for user,corr_list in self.corr.items():
            print(len(self.corr[user]))
            self.corr[user] = sorted(corr_list, key = lambda x: x[1], reverse=True)

    def predict_on_pair(self, user: int, item: int):
        if item != -1:
            if item == 409:
                print(item)
            k_nearest_users = [self.corr[user][i][0] for i in range(self.k)]
            return mode(k_nearest_users)
        else:
            return 3 # TODO - check this

    def upload_params(self):
        with open(CORRELATION_PARAMS_FILE_PATH, 'rb') as file:
            self.corr = pickle.load(file)

    def save_params(self):
        with open(CORRELATION_PARAMS_FILE_PATH, 'wb') as file:
            pickle.dump(self.corr, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    knn_config = Config(k=10)
    train, validation = get_data()
    train = train.iloc[:200000, :]
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
