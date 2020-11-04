import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, \
            ReLU, Flatten, MaxPooling2D


tf.random.set_seed(1234)
def f_embedding(img_size=28, embedding_dim=64):
    """
    Implements embedding function for the query set.
    
    Parameters
    ----------
    img_size: int
        Size of the input image. Images expected to have
        the width and height equal.
    embedding_dim: int
        Dimension of the embedding for one image.
        
    Returns
    -------
    tensorflow.python.keras.engine.sequential.Sequential
        Sequential tensorflow architecture.
    """
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
