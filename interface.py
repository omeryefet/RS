from numpy import sqrt, square, array


class Regressor:
    def __init__(self):
        raise NotImplementedError

    def fit(self, train):
        raise NotImplementedError

    def predict_on_pair(self, user, item) -> float:
        """given a user and an item predicts the ranking"""

    def calculate_rmse(self, data: array):
        e=0
        for row in data:
            user, item, rating = row
            unknown_user = user < 0 or user > 6040
            unknown_item = item < 0 or item > 3224
            if unknown_user or unknown_item:
                continue
            else:
                e += square(rating - self.predict_on_pair(user, item))
        return sqrt(e / data.shape[0])

