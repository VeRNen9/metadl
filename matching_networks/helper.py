import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, \
            ReLU, Flatten, MaxPooling2D


tf.random.set_seed(1234)


def embedding_architecture(input_set, embedding_dim):
    """
    Implements embedding function architecture.
    """
    x = input_set
    for layer in range(4):
        conv_params = {
            'filters': embedding_dim,
            'kernel_size': (3, 3),
            'padding': 'same'
        }
        x = Conv2D(**conv_params)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    return x


def cosine_distance(x, y):
    """
    Implements the cosine distance between two tensors.
    """
    return 1 - (-1 * tf.keras.losses.cosine_similarity(
        tf.expand_dims(x, 1), y))


def attention_mechanism(f, g, distance_function=cosine_distance):
    """
    Implements the attention mechanism. Computes the cosine distance
    between each image embedding in the query set and the embeddings
    of the support set images. Obtains a probability distribution
    for each image in the query set.
    
    Parameters
    ----------
    f: tf.Tensor
        Embeddings for the images in the query set.
    g: tf.Tensor
        Embeddings for the images in the support set.
    distance_func: function
        Function that implements the distance measure to be
        used for computing the similarities. Has to accept
        only the embeddings. If additional parameters needed
        consider implementing a partial.
        
    Returns
    -------
    tf.Tensor
    A probability distributions for each image in the query set.
    """
    return tf.nn.softmax(distance_function(f, g), name='attention_mechanism')
