
import numpy as np
import pandas as pd
from scipy import sparse
# from sklearn.model_selection import train_test_split


class Dataset:
    
    def __init__(self,
                 df,
                 user='user',
                 item='item',
                 rating='rating',
                 dtype=np.float32,
                 split_frac=0.05
                 ):

        self._user = user
        self._item = item
        self._rating = rating
        self.df = df
        self.dtype = dtype
        self.user_list = np.array([])
        self.item_list = np.array([])
    
    @property
    def user(self):
        return self.df[self._user]
    
    @property
    def item(self):
        return self.df[self._item]
    
    @property
    def rating(self):
        return self.df[self._rating]
        
    def get_factorizer(self, attr='user'):
        if attr not in ['user', 'item']:
            raise ValueError(f"Unknown attribute {attr}, only user, item was accepted")
        data = getattr(self, attr)
        return pd.factorize(data)
    
    def transform_to_matrix(self):
        user, self.user_list = self.get_factorizer("user")
        item, self.item_list = self.get_factorizer('item')
        n_user: int = len(self.user_list)
        n_item: int = len(self.item_list)
        return sparse.coo_matrix(
            (self.rating, (user, item)),
            shape=(n_user, n_item),
            dtype=self.dtype
        )
    
    def get_csr(self):
        return self.transform_to_matrix().tocsr()
    
    def get_csc(self):
        return self.transform_to_matrix().tocsc()
    
    def get_user_history(self, uid):
        res = {"user id": uid}
        history = self.get_csr()
        user_idx = np.where(self.user_list == uid)[0][0]
        list_idx_item = history[user_idx].indices
        item_id = self.item_list[list_idx_item]
        res['item id watched'] = item_id
        return res
    
    def get_item_history(self, iid):
        res = {"Item id": iid}
        history = self.get_csc()
        item_idx = np.where(self.item_list == iid)[0][0]
        list_idx_user = history[:, item_idx].indices
        user_id = self.user_list[list_idx_user]
        res['Users has been watch this item'] = user_id
        return res
