from interface import Regressor
from utils import get_data


class SimpleMean(Regressor):
    def __init__(self):
        self.user_means = {}

    def fit(self, X):
        raise NotImplementedError

    def predict_on_pair(self, user: int, item: int):
        raise NotImplementedError


if __name__ == '__main__':
    train, validation = get_data()
    baseline_model = SimpleMean()
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
