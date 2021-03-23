import numpy as np

from .Classification import Classification
from ..preprocessing.preprocessing import *


class KNN(Classification):
    def __init__(self, k_neighbors=3, preprocessing_function=pca()):
        super().__init__(name='knn')
        self.preprocessing_function = preprocessing_function
        self.k = k_neighbors

    def pre_processing(self, data):
        return self.preprocessing_function.transform(data)
    
    def predict(self, test):
        test = self.pre_processing(test)
        distances = np.linalg.norm(self._data - test, axis=1)
        idx_min = np.argsort(distances, axis=0)[:self.k]
        predicts = [self._labels[i] for i in idx_min]
        print('{} nearest neighbors {}'.format(self.k, dict(zip(idx_min, distances[idx_min]))))
        labels, count = np.unique(predicts, return_counts=True)
        if np.any(count[1:] == count[0]):
            return predicts[0]
        return labels[np.argmax(count)]

    def training(self, dataset, labels, input_shape, **kwargs):
        self.preprocessing_function.n_components = input_shape
        self._data = self.preprocessing_function.fit_transform(dataset)
        self._labels = labels

