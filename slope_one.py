
from interface import Regressor
from utils import get_data
import numpy as np
from config import *

class SlopeOne(Regressor):
    def __init__(self):
        self.popularity_differences = {}
        

    def fit(self, X: np.array):
        self.build_popularity_difference_dict(X)

    def build_popularity_difference_dict(self, data):
        for movie1 in data[USER_COL_NAME_IN_DATAEST]:
            for movie2 in data[ITEM_COL_NAME_IN_DATASET]:
                 
    def calc_difference(item1, item2): 
        '''
        item = {mid,rank}
        '''
                  
                
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
