import numpy as np
import tensorflow as tf


class Embedding(object):
    def __init__(self, node_size=None, embedding_dimension=None, W_values=None, use_truncated_normal_embedding=False):
        if node_size is None or embedding_dimension is None:
            return
        if W_values is None:
            W_values = self.initialize_weights(node_size=node_size, embedding_dimension=embedding_dimension,
                                               use_truncated_normal_weights=use_truncated_normal_embedding)
        self.E = W_values

    def initialize_weights(self, node_size, embedding_dimension, use_truncated_normal_weights):
        if not use_truncated_normal_weights:
            W_bound = np.sqrt(6. / embedding_dimension)
            W_values = tf.random_uniform([node_size, embedding_dimension], minval=-W_bound,
                                         maxval=W_bound, dtype=tf.float32)
        else:
            W_bound = 1.0 / np.sqrt(embedding_dimension)
            W_values = tf.truncated_normal([node_size, embedding_dimension], stddev=W_bound,
                                           dtype=tf.float32)
        return W_values
