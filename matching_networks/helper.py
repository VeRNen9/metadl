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


def cosine_distance(x, y):
    """
    Implements the cosine distance between two tensors.
    """
    return 1 - tf.keras.losses.cosine_similarity(x, y)


def attention_mechanism(f_embedding, g_embedding,
                        distance_func=cosine_distance):
    """
    Implements the attention mechanism. Computes the cosine distance
    between each image embedding in the query set and the embeddings
    of the support set images. Obtains a probability distribution
    for each image in the query set.
    
    Parameters
    ----------
    f_embeddings: tensorflow.python.framework.ops.EagerTensor
        Embeddings for the images in the query set.
    g_embeddings: tensorflow.python.framework.ops.EagerTensor
        Embeddings for the images in the support set.
    distance_func: function
        Function that implements the distance measure to be
        used for computing the similarities. Has to accept
        only the embeddings. If additional parameters needed
        consider implementing a partial.
        
    Returns
    -------
    tensorflow.python.framework.ops.EagerTensor
    A probability distributions for each image in the query set.
    """
    similarities = [distance_func(embedded_img, g_embedding)
                    for embedding_img in f_embedding]
    similarities = tf.stack(similarities)
    return tf.nn.softmax(similarities, axis=1)
