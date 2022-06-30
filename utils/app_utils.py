import pickle
import pandas as pd
import numpy as np
from ..models.dataset import Dataset
# from sqlalchemy import create_engine


# engine = create_engine("sqlite:///metadata")
RATING_PATH = "../data/rating.csv"


def load_data(path=RATING_PATH):
    df = pd.read_csv(path, index_col=0)
    return Dataset(df, user="UserID", item="MovieID", rating="Rating")


def get_user_history(user_id, dataset):
    if user_id not in dataset.user_list:
        raise ValueError(f"User id {user_id} not in dataset")
    return dataset.get_user_history(user_id)


def read_data(query, conn):
    return pd.read_sql_query(query, conn)


def get_movie_information(iid, table="movielens", sql_engine=engine):
    query = f"""SELECT * FROM {table} WHERE MovieID = {iid}"""
    return pd.read_sql_query(query, sql_engine)


def get_metadata(*items):
    res = []
    for item in items[0]:
        res.append(get_movie_information(item))
    return pd.concat(res, ignore_index=True)


def get_res_result(model, user_id, dataset, topk=10):
    if user_id not in dataset.user_list:
        raise ValueError(f"User id {user_id} not in dataset")
    idx = np.where(dataset.user_list == user_id)[0][0]
    item_id, score = model.recommend(idx, dataset.get_csr()[idx], N=topk)
    df = get_metadata(item_id)
    df['score'] = score
    return df

