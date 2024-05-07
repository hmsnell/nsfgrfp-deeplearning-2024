import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square
import numpy as np

class VAE(tf.keras.Model):
    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.hidden_dim = 20 # H_d # uhh
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ############################################################################################
        # TODO: Implement the fully-connected encoder architecture described in the notebook.      #
        # Specifically, self.encoder should be a network that inputs a batch of input images of    #
        # shape (N, 1, H, W) into a batch of hidden features of shape (N, H_d). Set up             #
        # self.mu_layer and self.logvar_layer to be a pair of linear layers that map the hidden    #
        # features into estimates of the mean and log-variance of the posterior over the latent    #
        # vectors; the mean and log-variance estimates will both be tensors of shape (N, Z).       #
        ############################################################################################
        # Replace "pass" statement with your code

        self.encoder = tf.keras.Sequential([

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(units = self.hidden_dim, activation = 'relu'),

            tf.keras.layers.Dense(units = self.hidden_dim, activation = 'relu') ,   

            tf.keras.layers.Dense(units = self.hidden_dim, activation = 'relu'), 

            ], name = "encoder")
        
        self.mu_layer = tf.keras.layers.Dense(self.latent_size)
        self.logvar_layer = tf.keras.layers.Dense(self.latent_size)


        ############################################################################################
        # TODO: Implement the fully-connected decoder architecture described in the notebook.      #
        # Specifically, self.decoder should be a network that inputs a batch of latent vectors of  #
        # shape (N, Z) and outputs a tensor of estimated images of shape (N, 1, H, W).             #
        ############################################################################################
        # Replace "pass" statement with your code
        
        self.decoder = tf.keras.Sequential([

            tf.keras.layers.Dense(self.hidden_dim, activation = 'relu'),

            tf.keras.layers.Dense(self.hidden_dim, activation = 'relu') ,   

            tf.keras.layers.Dense(self.hidden_dim, activation = 'relu'), 

            tf.keras.layers.Dense(2000, activation = "sigmoid"),

            tf.keras.layers.Reshape((1, 40, 50)) 

            ], name = "decoder")

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################


    def call(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ############################################################################################
        # TODO: Implement the forward pass by following these steps                                #
        # (1) Pass the input batch through the encoder model to get posterior mu and logvariance   #
        # (2) Reparametrize to compute  the latent vector z                                        #
        # (3) Pass z through the decoder to resconstruct x                                         #
        ############################################################################################
        # Replace "pass" statement with your code
        
        x_hat = self.encoder(x)
        mu = self.mu_layer(x_hat)
        logvar = self.logvar_layer(x_hat)
        latent_vector = reparametrize(mu, logvar)
        x_hat =self.decoder(latent_vector)

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar

def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with mean mu and
    standard deviation sigma, such that we can backpropagate from the z back to mu and sigma.
    We can achieve this by first sampling a random value epsilon from a standard Gaussian
    distribution with zero mean and unit variance, then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network, it helps to
    pass this function the log of the variance of the distribution from which to sample, rather
    than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns: 
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
    """
    z = None
    ################################################################################################
    # TODO: Reparametrize by initializing epsilon as a normal distribution and scaling by          #
    # posterior mu and sigma to estimate z                                                         #
    ################################################################################################
    # Replace "pass" statement with your code
    
    epsilon = tf.random.normal(shape = mu.shape, mean = 0, stddev = 1) 
    z = tf.sqrt(tf.exp(logvar)) * epsilon + mu 

    ################################################################################################
    #                              END OF YOUR CODE                                                #
    ################################################################################################
    return z


def kil_adavae_loss(x, x_hat, mu, gamma_val, logvar, is_winner):
    """
    Calculates the KIL-AdaVAE loss function.

    Args:
        x: Input data (e.g., sensor readings).
        x_decoded_mean: Mean of the reconstructed data distribution from the decoder.
        z_mean: Mean of the encoded latent distribution.
        z_log_var: Log variance of the encoded latent distribution.
        is_healthy: Boolean tensor indicating whether each sample is healthy or not.

    Returns:
        Tensor: The total loss value.
    """

    # MSE Reconstruction loss
    recon_loss =  tf.reduce_mean(tf.square(x - x_hat))
   
    # KL divergence
    kl_div = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1)

    #  winner focused KL div
    kil_loss = tf.reduce_mean(kl_div * tf.cast(is_winner, tf.float32))

    # Total loss
    total_loss = tf.reduce_mean(recon_loss + kl_div - gamma_val * kil_loss)

    return total_loss