
from interface import Regressor
from utils import get_data


class SlopeOne(Regressor):
    def __init__(self):
        self.popularity_differences = {}
        

    def fit(self, X: np.array):
       raise NotImplementedError

    def build_popularity_difference_dict(self, data):
        raise NotImplementedError

    def predict_on_pair(self, user: int, item: int):
        raise NotImplementedError

    def upload_params(self):
        raise NotImplementedError

    def save_params(self):
         raise NotImplementedError

    

if __name__ == '__main__':
    train, validation = get_data()
    slope_one = SlopeOne()
    slope_one.fit(train)
    print(slope_one.calculate_rmse(validation))
