import os
import logging
import csv 

import tensorflow as tf

from metadl.api.api import MetaLearner, Learner, Predictor
from helper import embedding_architecture, cosine_distance, attention_mechanism


# Line necessary for duranium. Comment it otherwise.
os.environ["CUDA_VISIBLE_DEVICES"] = "[1-7]"
tf.random.set_seed(1234)

@gin.configurable
class MyMetaLearner(MetaLearner):

    def __init__(self, img_size, num_channels, n_way=5, k_shot=1,
                 embedding_dim=64, distance_func='cosine'):
        super().__init__()
        self.img_size = img_size
        self.num_channels = num_channels
        self.n_way = n_way
        self.k_shot = k_shot
        self.embedding_dim = embedding_dim
        self.distance_func = self._set_distance_func(distance_func)
        
        self.f = embedding_architecture(
            img_size=img_size,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            name='f_embedding')
        
        self.g = embedding_architecture(
            img_size=img_size,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            name='g_embedding')
        
        self.attention = attention_mechanism(
            f_embedding=f,
            g_embedding=g,
            distance_func=self.distance_func)

        self.classifier = None
        
    def _set_distance_func(self, distance_func):
            if distance_func == 'cosine':
                self.distance_func = cosine_distance
            else:
                raise ValueError('Only cosine distance supported.')

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

