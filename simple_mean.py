from interface import Regressor
from utils import get_data


class SimpleMean(Regressor):
    def __init__(self):
        self.user_means = {}

    def fit(self, X):
        big_dict= (X.groupby(['user'])[["rating"]].mean().to_dict())
        self.user_means = big_dict["rating"]  #dict that holds the avg rating for each user


    def predict_on_pair(self, user: int, item: int):
        return self.user_means[user]


if __name__ == '__main__':
    train, validation = get_data()
    baseline_model = SimpleMean()
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
