import numpy as np
from sklearn.base import TransformerMixin
from sax import SAX

class SAXTransformer(TransformerMixin):
    def __init__(self, **kwargs):
        self.points_per_symbol = kwargs.get('points_per_symbol', SAX.default_points_per_symbol)
        self.a = kwargs.get('a', SAX.default_a)
        self.params = {'points_per_symbol': self.points_per_symbol,
                       'a': self.a}

    def transform(self, X):
        params = {}
        if self.a:
            params['a'] = self.a
        if self.points_per_symbol:
            params['points_per_symbol'] = self.points_per_symbol
        return np.array([SAX(i, **params).stringify() for i in X])


    def fit(self, X, y = None, **kwargs):
        return self


    def get_params(self, deep = True):
        return self.params
