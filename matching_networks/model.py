import os
import logging
import csv 

import tensorflow as tf

from metadl.api.api import MetaLearner, Learner, Predictor
from helper import f_embedding


# Line necessary for duranium. Comment it otherwise.
os.environ["CUDA_VISIBLE_DEVICES"] = "[1-7]"
tf.random.set_seed(1234)

@gin.configurable
class MyMetaLearner(MetaLearner):

    def __init__(self, img_size, n_way=5, k_shot=1,
                 embedding_dim=64,
                 distance_func=tf.losses.cosine_distance):
        super().__init__()
        self.img_size = img_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.embedding_dim = embedding_dim
        self.distance_func = distance_func
        
        self.f = f_embedding(img_size, embedding_dim)
        self.g = None
        self.attention = None
        self.classifier = None

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

