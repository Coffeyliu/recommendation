#THIS .PY NOW INCLUDE CALCULATION FOR BOTH UserUser and ItemItem CF

import pandas as pd
import numpy as np
from random import shuffle
import operator
import math

def train_test_seen_unseen(user_movie_table, user_num):
    """created train, test, seen, unseen index lists.
    Parameters
    --------------
    user_movie_table : pd.DataFrame
        table with user_movie rating info.
    user_num : Int
        user_num start from 1.

    Returns
    --------------
    train_index : list
        a list with the movie watched and rated by user_i. But without info which put in test_index.
    test_index : list
        a list has the true value but was hided.
    watched_index : list
        a list of index of watched movies.
    unseen_index : list
        a list of index of unseen movies.
        
    Note
    ---------------
    train_index and test_index are all from watched_index
    """
    mu = user_movie_table.T
    user1 = mu[user_num]
    
    watched_index = []
    unseen_index = []
    for i in range(len(user1)):
        if user1[i] >=0:
            watched_index.append(i)
        else:
            unseen_index.append(i)
    shuffle(watched_index)
    test_index = watched_index[:round(0.1*len(watched_index))]
    train_index_temp = list(map(lambda x: x if x not in test_index else np.nan, watched_index))
    train_index = list(pd.Series(train_index_temp).dropna().apply(int))
    
    return train_index, test_index, watched_index, unseen_index
    


def weight_calculator(table, index1, index2):
    """find pearson corr between two users/items.
    
    Parameters
    --------------
    user_movie_table : pd.DataFrame
        user_movie rating info.
    index1, index2 : Integer
        pointer in table. Index for users.
    
    Note
    ---------------
    if input table is user_movie_table, the function will calculate USERs' correlation scores;
    if input table is movie_user_table, the function will calculate ITEMs' correlation scores.
       
    """
    user_movie_table = table 
    
    mu = user_movie_table.T
    
    user1 = mu[index1]
    user2 = mu[index2]
    common = mu[[index1,index2]].dropna()

    if len(common) < 5 or len(common) is None:
        return 0
    else:
        user1_watched = user1.dropna()
        user2_watched = user2.dropna()
        
        x1_mean = user1.mean()
        x2_mean = user2.mean()
        numerator = np.array(list(map(lambda x,y : (x-x1_mean)*(x-x2_mean), common[index1],common[index2]))).sum()
        
        left_corner = np.array(list(map(lambda x : (x-x1_mean)**2, user1_watched))).sum()**0.5
        right_corner = np.array(list(map(lambda x : (x-x2_mean)**2, user2_watched))).sum()**0.5
        denominator = left_corner * right_corner
        
        weight_between_two_users = numerator/denominator
        return weight_between_two_users


def pred_specific_movie_score(user_movie_table, user_num, movie_num, k_nearest=20):
    """Using k_nearest neighbor to predict one user score towards a specific movie.
    
    Parameters
    ------------
    user_movie_table : pd.DataFrame
        table with user_movie rating info.
    user_num : Int
        user_num starts from 1.
    movie_num : Int
        movie_num starts from 1.
    k_nearest : Int
        number of nearest users
    
    Returns
    ------------
    score_1_3 : Int
        ex: weight between user1 and user3
    """
    
    movie_index = movie_num - 1
    mu = user_movie_table.T

    #1.find people who rates movie3
    movie3_rating = user_movie_table.ix[:,movie_index]
    movie3_rating = movie3_rating.dropna() #so I find the movie3 all user ratings
    
    #2.get users who have rated movie3
    movieId = pd.DataFrame(user_movie_table.ix[user_num,:]).reset_index()['movieId']
    users_movie3 = list(pd.DataFrame(movie3_rating.reset_index())['userId'])
    movie3_rating_score = list(pd.DataFrame(movie3_rating.reset_index())['rating'][movieId[movie_index]])
    
    #3.find K nearest users
    weight_1_users = {}
    for i in range(len(users_movie3)):
        weight_1_i = weight_calculator(user_movie_table, user_num, users_movie3[i])
        weight_1_users[users_movie3[i]] = weight_1_i
    sorted_weight_1_users = sorted(weight_1_users.items(), key=operator.itemgetter(1), reverse=True)
    sorted_k_nearest_users = list(item[0] for item in sorted_weight_1_users)[:k_nearest]
    sorted_k_nearest_weights = list(item[1] for item in sorted_weight_1_users)[:k_nearest]
    
    #4.calculate predict score for user1 of movie3
    numerator_list = []
    denominator_list = []
    for i in range(len(sorted_k_nearest_users)):
        weight_1_i = sorted_k_nearest_weights[i]

        useri_watched = mu[users_movie3[i]].dropna()
        xi_mean = useri_watched.mean()

        score_i = movie3_rating_score[i]
        numerator_i = weight_1_i * (score_i - xi_mean)
        numerator_list.append(numerator_i)

        denominator_i = (weight_1_i**2)**0.5
        denominator_list.append(denominator_i)
    
    numerator = np.array(numerator_list).sum()
    denominator = np.array(denominator_list).sum()
    
    user1_watched = mu[user_num].dropna()
    x1_mean = user1_watched.mean()
    score_1_3 = x1_mean + numerator/denominator
    
    return score_1_3



def pred_series_movie_scores(table, user_num, unseen_index, k_nearest=20):
   
    """predict the score for user_i to a series of his unseen movies.
    Parameters
    ------------
    table : pd.DataFrame
        if input table is user_movie_table, the CF is for User-User;
        
    user_num : Int
        user_num start from 1.
    unseen_index : list
        a list of index of unseen movies.
    k_nearest : Int
        number of nearest users. Default=20

    Returns
    -------------
    recommend_movieId : list
        a descent sorted list of movieId to watch
    recommend_movie_score : list
        a descent sorted list of predicted score user_i will give     
    """
    user_movie_table = table
    
    movie_pred_score = {}
    for i in range(len(unseen_index)):
        score_i = pred_specific_movie_score(user_movie_table, 1, unseen_index[i], k_nearest)
        movie_pred_score[unseen_index[i]] = score_i
        sorted_movie_pred_score = sorted(movie_pred_score.items(), key=operator.itemgetter(1), reverse=True)
    
    recommend_movie_index = list(map(lambda tp: tp[0],sorted_movie_pred_score))
    recommend_movie_score = list(map(lambda tp: tp[1],sorted_movie_pred_score))
    
    movieId = pd.DataFrame(user_movie_table.ix[user_num,:]).reset_index()['movieId']
    recommend_movieId = [movieId[i] for i in recommend_movie_index]
    
    return recommend_movieId, recommend_movie_score


def raw_to_pred(x):
    """round number in half
    ex: 3.76 will be rounded to 4, and 3.744 will be rounded to 3.5.
    """
    middle = (math.floor(x) + math.ceil(x))/2
    
    if x < (middle - 0.25):
        return math.floor(x)
    
    if x >= (middle - 0.25) and x < (middle + 0.25):
        return middle
    
    if x >= (middle + 0.25):
        return math.ceil(x)

