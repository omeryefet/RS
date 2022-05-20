
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

        """
        # Find all movies user U ranked
        # For item1, item 2, if item1 and item 2 were ranked by the user, include them in the vector
        # If (item1_id, item2_id) not in dict:
        #   calc difference for the 2 vectors
        #   Store the result as (item1,item2) = result
        #   Store the opposite result as (item2_id, item1_id) = -result
        """
        # a dictionary of userid: movie_id of movies the user ranked
        user_ranked_movies = {userid: data[ITEM_COL_NAME_IN_DATASET].where(userid)
                              for userid in data[USER_COL_NAME_IN_DATAEST]}

        for movie_1 in data[ITEM_COL_NAME_IN_DATASET]:
            for movie2 in data[ITEM_COL_NAME_IN_DATASET]:
                if movie_1 in user_ranked_movies[userid]:
                    pass
                 
    def calc_difference(self, item1: np.array, item2: np.array):
        """
        Item is a vector of all its' ranking
        """
        # drop rows with nan
        return (item1 - item2) / len(item1)




                
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
