import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, \
            ReLU, Flatten, MaxPooling2D


tf.random.set_seed(1234)
def f_embedding(img_size=28, embedding_dim=64):
    model = Sequential()
    
    model.add(Conv2D(embedding_dim, (3, 3), padding='same',
                     input_shape=(img_size, img_size, 3)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(embedding_dim, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(embedding_dim, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(embedding_dim, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    
    model._name = 'F_embedding'

    return model
