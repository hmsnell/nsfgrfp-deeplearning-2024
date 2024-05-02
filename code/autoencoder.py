import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square
from preprocessing import * 
import numpy as np
from keras.layers import Input, Dense, Layer
from keras.models import Model
from keras import backend as K

class AE(tf.keras.Model):
    def __init__(self, vocab_size, latent_size = 15, embed_size = 64):
        super(AE, self).__init__()
        self.vocab_size = vocab_size
        self.latent_size = latent_size  # Z
        self.hidden_dim = 260  # H_d
        self.embed_size = embed_size
        self.encoder = None
        self.embed_size = None
        self.decoder = None

        # embedding layer 
        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.vocab_size, output_dim = self.embed_size)

        # encoder structure 
        self.encoder = tf.keras.Sequential([ 
            
            tf.keras.layers.Dense(units = self.vocab_size, activation = 'relu'),

            tf.keras.layers.Dense(units = self.hidden_dim, activation = 'relu'),

            tf.keras.layers.Dense(units = self.latent_size, activation = 'relu')

        ], name = "encoder")
        
        # decoder structure 

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.latent_size, activation = "relu"),  

            tf.keras.layers.Dense(self.hidden_dim, activation = "relu"), 

            tf.keras.layers.Dense(self.vocab_size, activation = "sigmoid")

            # potentially need a reshape here to get back embedding matrix dims 
            
        ], name = "decoder")

    def call(self, x):
        
        #rint(x.shape())
        #embedding = self.embedding_layer(x)
        x_hat = self.encoder(x)
        x_out = self.decoder(x_hat)

        return x_out

class ClusteringLayer(Layer): # reference: https://github.com/Tony607/Keras_Deep_Clustering/blob/master/Keras-DEC.ipynb
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, input_dim))]
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim),
                                        initializer='glorot_uniform',
                                        name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters


def bce_function(x_hat, x):

    bce_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.SUM,
    )
    reconstruction_loss = bce_fn(x, x_hat) * x.shape[
        -1]  # Sum over all loss terms for each data point. This looks weird, but we need this to work...
    return reconstruction_loss