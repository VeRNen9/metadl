import csv
import gin
import os
import logging

import tensorflow as tf

from metadl.api.api import MetaLearner, Learner, Predictor
from .helper import embedding_architecture, cosine_distance, attention_mechanism


# Line necessary for duranium. Comment it otherwise.
os.environ["CUDA_VISIBLE_DEVICES"] = "[1-7]"
tf.random.set_seed(1234)

@gin.configurable
class MyMetaLearner(MetaLearner):
    
    distance_functions = {
        'cosine': cosine_distance
    }

    def __init__(self, img_size, num_channels, n_way=5, k_shot=1,
                 embedding_dim=64, distance_func='cosine'):
        super().__init__()
        self.img_size = img_size
        self.num_channels = num_channels
        self.n_way = n_way
        self.k_shot = k_shot
        self.embedding_dim = embedding_dim
        self.distance_func = self._set_distance_func(distance_func)
        
        self.model = self._matching_architecture()

    def _set_distance_func(self, distance_func):
        if distance_func not in MyMetaLearner.distance_functions:
            raise ValueError(
                'Distance functions not supported.\n' +
                'Available functions: ' +
                ', '.join(list(MyMetaLearner.distance_functions.keys())))
        return MyMetaLearner.distance_functions[distance_func]

    def _matching_architecture(self):
        support_set = tf.keras.Input(
            shape=(self.img_size, self.img_size, self.num_channels, ))
        query_set = tf.keras.Input(
            shape=(self.img_size, self.img_size, self.num_channels, ))

        f = embedding_architecture(query_set,
                                   embedding_dim=self.embedding_dim)
        g = embedding_architecture(support_set,
                                   embedding_dim=self.embedding_dim)
        
        attention = attention_mechanism(
            f, g, distance_func=self.distance_func)
        
        model = tf.keras.Model(inputs=[support_set, query_set],
                               outputs=attention)
        model._name = 'matching_networks'
        return model

    def meta_fit(self, meta_dataset_generator) -> Learner:
        raise NotImplemented()


class MyLearner(Learner):

    def __init__(self):
        super().__init__()

    def fit(self, dataset_train) -> Predictor:
        raise NotImplemented()

    def save(self, model_dir):
        raise NotImplemented()
            
    def load(self, model_dir):
        raise NotImplemented()

    
class MyPredictor(Predictor):

    def __init__(self):
        super().__init__()

    def predict(self, dataset_test):
        raise NotImplemented()

