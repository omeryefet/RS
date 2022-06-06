from interface import Regressor
from utils import Config, get_data

import numpy as np
np.random.seed(0)
from tqdm import tqdm


class MatrixFactorization(Regressor):
    def __init__(self, config):
        self.lr = config.lr # learning rate
        self.gamma = config.gamma # gamma (regularization parameter)
        self.epochs = config.epochs # number of epoch to train
        self.k = config.k # holds the number of dimensions of the new vectors
        self.pu = None # p_u matrix (users)
        self.qi = None # q_i matrix (items)
        self.n_users = None # holds the number of users
        self.n_items = None # holds the number of items
        self.user_biases = None # b_u vector (users)
        self.item_biases = None # b_i vector (items)
        self.current_epoch = 1 # holds the number of the current epoch
        self.mu = None # holds the mean of all ratings
        self.q_mul_p = None # holds the multiple of qi and pu

    def record(self, covn_dict): # print the values of 3 measures
        epoch = '{:02d}'.format(self.current_epoch)
        temp = f'| epoch # {epoch} :'
        for key, value in covn_dict.items():
            key = f'{key}'
            val = '{:.4}'.format(value)
            result = '{:<25}'.format(f'  {key} : {val}')
            temp += result
        print(temp)

    def calc_regularization(self): # calculate the regularization of pu,qi,bu,bi
        sum_p_users_square = np.sum(self.pu ** 2)
        sum_q_items_square = np.sum(self.qi ** 2)
        sum_biases_users_square = np.sum(self.user_biases ** 2)
        sum_biases_items_square = np.sum(self.item_biases ** 2)
        return self.gamma * (sum_biases_users_square + sum_biases_items_square + sum_p_users_square + sum_q_items_square)

    def fit(self, X):
        self.n_users = len(np.unique(X[:,0])) # number of unique users
        self.n_items = len(np.unique(X[:,1])) # number of unique items
        self.mu = np.mean(X[:,2]) # mean of ratings
        self.user_biases = np.zeros(self.n_users) # initializing a vector of zeroes (users)
        self.item_biases = np.zeros(self.n_items) # initializing a vector of zeroes (items)
        self.pu = np.random.rand(self.n_users, self.k) # initializing a matrix with random values (number of unique users X k)
        self.qi = np.random.rand(self.k, self.n_items) # initializing a matrix with random values (k X number of unique items)
        while self.current_epoch <= self.epochs: # run over number of epochs
            self.run_epoch(X)
            train_rmse = self.calculate_rmse(X) # calculate the RMSE
            train_mse = np.square(train_rmse) # calculate the MSE
            train_objective = train_mse * X.shape[0] + self.calc_regularization() # calculate the objective
            epoch_convergence = {"Train RMSE": train_rmse, "Train MSE": train_mse, "Train Objective": train_objective}
            self.record(epoch_convergence)
            self.current_epoch += 1

    def run_epoch(self, data: np.array):
        for row in data:
            user, item, rating = row
            qi_pu = self.pu[user, :].dot(self.qi[:, item])
            error = rating - (self.mu + self.user_biases[user] + self.item_biases[item] + qi_pu) # calculate the error of prediction
            self.user_biases[user] += self.lr * (error - self.gamma * self.user_biases[user]) # updating bu for a specific user
            self.item_biases[item] += self.lr * (error - self.gamma * self.item_biases[item]) # updating bi for specific item
            self.pu[user, :] += self.lr * (error * self.qi[:, item] - self.gamma * self.pu[user, :]) # updating pu for specific user and k dimensions
            self.qi[:, item] += self.lr * (error * self.pu[user, :] - self.gamma * self.qi[:, item]) # updating qi for specific item and k dimensions
        self.q_mul_p = self.pu.dot(self.qi)


    def predict_on_pair(self, user, item):
        if item == -1: # if the item does not exist on the train set
            predict = self.mu + self.user_biases[user] # mean rating + user bias
            if predict > 5:
                return 5
            elif predict < 0:
                return 0
            else:
                return predict
        else: # return the mean rating + user bias + item bias + value of user-item multiplied vector
            predict = self.mu + self.user_biases[user] + self.item_biases[item] + self.q_mul_p[user, item] # prediction (for specific user and specific item)
            if predict > 5:
                return 5
            elif predict < 0:
                return 0
            else:
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
    print(baseline_model.calculate_rmse(validation))
