from interface import Regressor
from utils import Config, get_data

import numpy as np
np.random.seed(0)
from tqdm import tqdm


class MatrixFactorization(Regressor):
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.epochs = config.epochs
        self.k = config.k
        self.pu = None                                    # p_u (users) matrix
        self.qi = None                                    # q_i (items) matrix
        self.n_users = None                               # will hold number of unique users
        self.n_items = None                               # will hold number of unique items
        self.user_biases = None                           # b_u (users) vector
        self.item_biases = None                           # b_i (items) vector
        self.current_epoch = 1
        self.global_bias = None                           # will hold mean of all ratings
        self.q_mul_p = None

    def record(self, covn_dict):
        epoch = '{:02d}'.format(self.current_epoch)
        temp = f'| epoch # {epoch} :'
        for key, value in covn_dict.items():
            key = f'{key}'
            val = '{:.4}'.format(value)
            result = '{:<25}'.format(f'  {key} : {val}')
            temp += result
        print(temp)

    def calc_regularization(self):
        sum_biases_users_square = sum(self.user_biases ** 2)
        sum_biases_items_square = sum(self.item_biases ** 2)
        sum_p_users_square = sum(sum(self.pu ** 2))
        sum_q_items_square = sum(sum(self.qi ** 2))
        return self.gamma * (sum_biases_users_square + sum_biases_items_square + sum_p_users_square + sum_q_items_square)

    def fit(self, X):
        self.n_users = len(np.unique(X[:,0]))
        self.n_items = len(np.unique(X[:,1]))
        self.global_bias = np.mean(X[:,2])
        self.user_biases = np.zeros(self.n_users)         # initializing to a vector with zeroes (length = num of unique users)
        self.item_biases = np.zeros(self.n_items)         # initializing to a vector with zeroes (length = num of unique items)
        self.pu = np.random.rand(self.n_users, self.k)    # initializing to a matrix with random values (num of unique users as rows, k as columns)
        self.qi = np.random.rand(self.k, self.n_items)    # initializing to a matrix with random values (k as rows, num of unique items as columns)
        while self.current_epoch <= self.epochs:
            self.run_epoch(X)
            train_rmse = self.calculate_rmse(X)
            train_mse = np.square(train_rmse)
            train_objective = train_mse * X.shape[0] + self.calc_regularization()
            epoch_convergence = {"Train RMSE": train_rmse, "Train MSE": train_mse, "Train Objective": train_objective}
            self.record(epoch_convergence)
            self.current_epoch += 1

    def run_epoch(self, data: np.array):
        for row in data:
            user, item, rating = row
            qi_pu = self.pu[user, :].dot(self.qi[:, item])
            error = rating - (self.global_bias + self.user_biases[user] + self.item_biases[item] + qi_pu)
            self.user_biases[user] += self.lr * (error - self.gamma * self.user_biases[user])                  # updating bu for specific user
            self.item_biases[item] += self.lr * (error - self.gamma * self.item_biases[item])                  # updating bi for specific item
            self.pu[user, :] += self.lr * (error * self.qi[:, item] - self.gamma * self.pu[user, :])           # updating pu for specific user and all k dimensions
            self.qi[:, item] += self.lr * (error * self.pu[user, :] - self.gamma * self.qi[:, item])           # updating qi for specific item and all k dimensions
        self.q_mul_p = self.pu.dot(self.qi)


    def predict_on_pair(self, user, item):
        if item == -1:
            return self.global_bias + self.user_biases[user]
        else:
            return self.global_bias + self.user_biases[user] + self.item_biases[item] + self.q_mul_p[user, item] # prediction (for specific user and specific item)


if __name__ == '__main__':
    baseline_config = Config(
        lr=0.01,
        gamma=0.001,
        k=24,
        epochs=10)

    train, validation = get_data()
    baseline_model = MatrixFactorization(baseline_config)
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
