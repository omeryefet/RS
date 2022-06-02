from interface import Regressor
from utils import get_data
from config import *
import pandas as pd
import pickle

class SimpleMean(Regressor):
    def __init__(self):
        self.user_means = {}

    def fit(self, X):
        Y = pd.DataFrame(X, columns = [USER_COL,ITEM_COL,RATING_COL])
        self.user_means = Y.groupby([USER_COL])[RATING_COL].mean().to_dict()
        self.save_params()

    def predict_on_pair(self, user: int, item: int):
        return self.user_means[user]

    def save_params(self):
        with open(SIMPLE_MEAN_PATH, 'wb') as file:
            pickle.dump(self.user_means, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    train, validation = get_data()
    baseline_model = SimpleMean()
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))