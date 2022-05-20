import numpy as np

from interface import Regressor
from utils import get_data, Config

class KnnUserSimilarity(Regressor):
    def __init__(self, config):
        raise NotImplementedError

    def fit(self, X: np.array):
        raise NotImplementedError

    def build_item_to_itm_corr_dict(self, data):
        raise NotImplementedError

    def predict_on_pair(self, user: int, item: int):
        raise NotImplementedError

    def upload_params(self):
        raise NotImplementedError


if __name__ == '__main__':
    knn_config = Config(k=10)
    train, validation = get_data()
    train = train[:200000, :]
    knn = KnnUserSimilarity(knn_config)
    knn.fit(train)
    print(knn.calculate_rmse(validation))
