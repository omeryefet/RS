from interface import Regressor
from utils import Config, get_data
import numpy as np
import math


class MatrixFactorization(Regressor):
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.train_epochs = config.epochs
        self.k = config.k                                 # will hold number of dimensions (features)
        self.pu = None                                    # p_u (users) matrix
        self.qi = None                                    # q_i (items) matrix
        self.n_users = None                               # will hold number of unique users
        self.n_items = None                               # will hold number of unique items
        self.user_biases = None                           # b_u (users) vector
        self.item_biases = None                           # b_i (items) vector
        self.current_epoch = 0
        self.global_bias = None                           # will hold mean of all ratings
        self.q_mul_p = None

    def record(self, covn_dict):
        epoch = "{:02d}".format(self.current_epoch)
        temp = f"| epoch   # {epoch} :"
        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.4}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)

    def calc_regularization(self):
        sum_b_u2 = sum(self.user_biases ** 2)
        sum_b_i2 = sum(self.item_biases ** 2)
        sum_q_i2 = sum(sum(self.qi**2))
        sum_p_u2 = sum(sum(self.pu**2))
        return self.gamma * (sum_b_u2 + sum_b_i2 + sum_q_i2 + sum_p_u2)

    def fit(self, X):
        self.n_users = X["user"].nunique()
        self.n_items = X["item"].nunique()
        self.global_bias = X["rating"].mean()
        self.user_biases = np.zeros(self.n_users)         # initializing to a vector with zeroes (length = num of unique users)
        self.item_biases = np.zeros(self.n_items)         # initializing to a vector with zeroes (length = num of unique items)
        self.qi = np.random.rand(self.k, self.n_items)    # initializing to a matrix with random values (k as rows, num of unique items as columns)
        self.pu = np.random.rand(self.n_users, self.k)    # initializing to a matrix with random values (num of unique users as rows, k as columns)
        while self.current_epoch < self.train_epochs:
            self.run_epoch(X.to_numpy())
            train_mse = np.square(self.calculate_rmse(X.to_numpy()))
            train_objective = train_mse * X.shape[0] + self.calc_regularization()
            epoch_convergence = {"train_objective": train_objective,
                                 "train_mse": train_mse}
            self.record(epoch_convergence)
            self.current_epoch += 1

    def run_epoch(self, data: np.array):
        for row in data:
            user, item, rating = row
            qi_mul_pu = self.pu[user, :].dot(self.qi[:, item])
            e_ui = rating - (self.global_bias + self.user_biases[user] + self.item_biases[item]+qi_mul_pu)    # prediction error (for specific user and specific item)
            self.user_biases[user] += self.lr * (e_ui - self.gamma * self.user_biases[user])                  # updating bu for specific user
            self.item_biases[item] += self.lr * (e_ui - self.gamma * self.item_biases[item])                  # updating bi for specific item
            self.qi[:, item] += self.lr * (e_ui * self.pu[user, :] - self.gamma * self.qi[:, item])           # updating qi for specific item and all k dimensions
            self.pu[user, :] += self.lr * (e_ui * self.qi[:, item] - self.gamma * self.pu[user, :])           # updating pu for specific user and all k dimensions
        self.q_mul_p = self.pu.dot(self.qi)

    def predict_on_pair(self, user, item):
        user = int(user)
        if math.isnan(item):                                                       # If the item is nan predict only with thhe user bias and the global bias
            return self.global_bias + self.user_biases[user]
        else:
            item = int(item)
            predict = self.global_bias + self.user_biases[user] + self.item_biases[item] + self.q_mul_p[user, item] # prediction (for specific user and specific item)
            return predict


if __name__ == '__main__':
    baseline_config = Config(
        lr=0.01,
        gamma=0.001,
        k=24,
        epochs=10)

    train, validation = get_data()
    baseline_model = MatrixFactorization(baseline_config)
    baseline_model.fit(train)
    print (baseline_model.calculate_rmse(validation))

