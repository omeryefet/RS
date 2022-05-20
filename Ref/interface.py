from numpy import sqrt, square, array


class Regressor:
    def __init__(self):
        raise NotImplementedError

    def fit(self, train):
        raise NotImplementedError

    def predict_on_pair(self, user, item) -> float:
        raise NotImplementedError

    def calculate_rmse(self, data: array):
        e = 0
        for row in data:
            user, item, rating = row
            e += square(rating - self.predict_on_pair(user, item))
        return sqrt(e / data.shape[0])
