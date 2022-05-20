from typing import Dict
import numpy as np
from config import BASELINE_PARAMS_FILE_PATH
from interface import Regressor
from utils import Config, get_data
import pickle
import math

class Baseline(Regressor):
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.train_epochs = config.epochs
        self.n_users = None
        self.n_items = None
        self.user_biases = None  # b_u (users) vector
        self.item_biases = None  # # b_i (items) vector
        self.current_epoch = 0
        self.global_bias = None

    def record(self, covn_dict: Dict):
        epoch = "{:02d}".format(self.current_epoch)
        temp = f"| epoch   # {epoch} :"
        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.4}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)

    def calc_regularization(self):
        b_u2 = [x ** 2 for x in self.user_biases]
        sum_b_u2 = sum(b_u2)
        b_i2 = [x ** 2 for x in self.item_biases]
        sum_b_i2 = sum(b_i2)
        return self.gamma * (sum_b_u2 + sum_b_i2)

    def fit(self, X):
        self.n_users = X["user"].nunique()                       #get the number of users
        self.n_items = X["item"].nunique()                       #get the number of items
        self.user_biases = np.zeros(self.n_users)                #initialize b_u (users) vector
        self.item_biases = np.zeros(self.n_items)                #initialize b_i (item) vector
        self.global_bias = X["rating"].mean()                    #get the mean of all users
        while self.current_epoch < self.train_epochs:
            self.run_epoch(X.to_numpy())
            train_mse = np.square(self.calculate_rmse(X.to_numpy()))
            train_objective = train_mse * X.shape[0] + self.calc_regularization()
            epoch_convergence = {"train_objective": train_objective,
                                 "train_mse": train_mse}
            self.record(epoch_convergence)
            self.current_epoch += 1
        self.save_params()

    def run_epoch(self, data: np.array):
        for row in data:                                         #in each epoch we update the deriveds
            user, item, rating = row
            difference_in_prediction = (rating - self.global_bias - self.user_biases[user] - self.item_biases[item])
            self.user_biases[user] += self.lr*(2*(difference_in_prediction)+self.gamma*2*self.user_biases[user])
            self.item_biases[item] += self.lr*(2*(difference_in_prediction)+self.gamma*2*self.item_biases[item])

    def predict_on_pair(self, user: int, item):
        user = int(user)
        if math.isnan(item):
            return self.global_bias + self.user_biases[user]    # If the item is nan predict with not his bias
        else:
            item = int(item)
            return self.global_bias + self.user_biases[user] + self.item_biases[item]

    def save_params(self):
        with open(BASELINE_PARAMS_FILE_PATH, 'wb') as f:          #save all the relevant parametes as pikle file
            pickle.dump([self.user_biases, self.item_biases, [self.global_bias]], f)


if __name__ == '__main__':
    baseline_config = Config(
        lr=0.001,
        gamma=0.001,
        epochs=10)

    train, validation = get_data()
    baseline_model = Baseline(baseline_config)
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
