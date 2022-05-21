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
import scipy.stats as st
import warnings
warnings.filterwarnings("error")


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
            # self.save_params()

    def build_item_to_itm_corr_dict(self, data):
        for i in tqdm(range(self.matrix.shape[1])):
            for j in range(i+1, self.matrix.shape[1]):
                if i != j:
                    item1 = np.array(self.matrix[:, i], dtype=np.int16)
                    item2 = np.array(self.matrix[:, j], dtype=np.int16)
                    bool_vector = np.array(item1 * item2) > 0
                    item1 = item1[bool_vector]
                    item2 = item2[bool_vector]
                    # print('item1  ',item1)
                    # print('item2  ',item2)
                    # print(np.corrcoef(item1,item2)[0,1])
                    if not any(item1 * item2):
                        items_corr = 0
                    elif np.var(item1) == 0 and np.var(item2) == 0:
                        items_corr = 1
                    elif np.var(item1) == 0 or np.var(item2) == 0:
                        items_corr = 1
                    elif any(item1 * item2):
                        # items_corr = np.random.uniform(-1,1)
                        try:
                            items_corr = np.corrcoef(item1,item2)[0,1]
                        except RuntimeWarning:
                            # print('item1  ', item1)
                            # print('item2  ', item2)
                            import ipdb
                            ipdb.set_trace()
                        # items_corr = st.pearsonr(item1,item2)[0]
                        #items_corr = 1
                    else:
                        items_corr = 0
                    self.corr[i].append((j, items_corr))
                    self.corr[j].append((i, items_corr))
        for item, corr_list in self.corr.items():
            self.corr[item] = sorted(corr_list, key = lambda x: x[1], reverse=True)

    def predict_on_pair(self, user: int, item: int):
        if item != -1:
            # if item == 409:
            #     print(user)
            #     print(item)
            #     x = self.corr[user]
            #     k_nearest_users = []
            #     for i in range(self.k):
            #         print(self.corr[user])
            #         z = self.corr[user][i]
            #         print(z[0])
            items_lst = []
            ranked_idxs = set(np.arange(len(self.matrix.shape[1]))[self.matrix[user,:] > 0])
            for i in self.corr[item]:
                if i[0] in ranked_idxs:
                    items_lst.append(self.matrix[user,i[0]])
                if len(items_lst) == self.k:
                    break
            return mode(items_lst)
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
