
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseEncoder:

    def _fit(self, data):
        pass

    def fit(self, data, return_type="sparse"):
        array = self._fit(data)
        if return_type == 'sparse':
            return csr_matrix(array)
        elif return_type == 'dataframe':
            cols = self.classes
            idx = self.index
            return pd.DataFrame(array, columns=cols, index=idx)
        else:
            return array

    def __call__(self, data, return_type='sparse'):
        return self.fit(data, return_type)


class TfidfEncoder(BaseEncoder):

    def __init__(self, **kwargs):
        self.engine = TfidfVectorizer(max_features=100)

    def _fit(self, data):
        arr = self.engine.fit_transform(data)
        self.classes = [f"col_{x}" for x in range(self.engine.max_features)]
        if not hasattr(data, "index"):
            self.index = list(range(len(data)))
        else:
            self.index = data.index.copy()
        return arr


class CategoricalEncoder(BaseEncoder):

    def __init__(self, **kwargs):
        self.engine = LabelBinarizer()

    def _fit(self, data):
        arr = self.engine.fit_transform(data)
        self.classes = self.engine.classes_
        if not hasattr(data, "index"):
            self.index = list(range(len(data)))
        else:
            self.index = data.index.copy()
        return arr


class ListEncoder(BaseEncoder):

    def __init__(self, **kwargs):
        self.engine = MultiLabelBinarizer()

    def _fit(self, data):
        arr = self.engine.fit_transform(data)
        self.classes = self.engine.classes_
        if not hasattr(data, "index"):
            self.index = list(range(len(data)))
        else:
            self.index = data.index.copy()
        return arr


class ContentBased:

    def __init__(self):
        self.n_feartures = 0
        self.list_features = []
        self.cate_features = []
        self.text_features = []
        # self.similarity = similarity
        self.simi_matrix = np.array([])
        self.categorical_encoder = CategoricalEncoder()
        self.text_encoder = TfidfEncoder()
        self.list_encoder = ListEncoder()
        self.item_index_map = {}
        self.item_id = []

    @staticmethod
    def cosine_similarity(arr):
        return cosine_similarity(arr)

    def recommend(self, iid, k=10):
        # Check if model not fitted
        if self.simi_matrix.size == 0:
            raise NotImplementedError(f"Model not fit yet")

        idx = self.item_index_map.get(iid)
        if idx is None:
            return {}
        item_vector = self.simi_matrix[idx]
        top = np.argsort(item_vector)[::-1][:k]
        item_rec = [self.item_id[x] for x in top]
        return (np.array(item_rec), item_vector[top])

    def fit(self, data, config):
        self.item_id = data[config['id_col']]
        self.item_index_map = {v: k for k, v in self.item_id.to_dict().items()}
        self.text_features = config['text_cols']
        self.cate_features = config['categorical_cols']
        self.list_features = config['list_cols']
        self.n_feartures = len(self.text_features) + \
                           len(self.cate_features) + \
                           len(self.list_features)

        self.simi_matrix = np.zeros((len(self.item_id), len(self.item_id)))

        if self.text_features:
            for cols in self.text_features:
                print(f"Compute similarity of feature {cols}")
                s_arr = self.cosine_similarity(self.text_encoder(data[cols]))
                self.simi_matrix += s_arr
        if self.cate_features:
            for cols in self.cate_features:
                print(f"Compute similarity of feature {cols}")
                s_arr = self.cosine_similarity(self.categorical_encoder(data[cols]))
                self.simi_matrix += s_arr
        if self.list_features:
            for cols in self.list_features:
                print(f"Compute similarity of feature {cols}")
                s_arr = self.cosine_similarity(self.list_encoder(data[cols]))
                self.simi_matrix += s_arr

        self.simi_matrix /= self.n_feartures
        print(f"Done train model, total {self.n_feartures} has trained")

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
