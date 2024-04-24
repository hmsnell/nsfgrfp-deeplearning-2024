import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square


class VAE(tf.keras.Model):
    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.hidden_dim = 260  # H_d
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

        # encoder structure 

        self.encoder = tf.keras.Sequential([ 
            tf.keras.layers.Flatten(),
            
            #tf.keras.layers.Dense(input_shape = self.input_size, units = self.hidden_dim, activation = 'relu'),
            tf.keras.layers.Dense(units = self.hidden_dim, activation = 'relu'),


            #tf.keras.layers.Dense(input_shape = self.hidden_dim, units = self.hidden_dim, activation = 'relu'),
            tf.keras.layers.Dense(units = self.hidden_dim, activation = 'relu'),


            #tf.keras.layers.Dense(input_shape = self.hidden_dim, units = self.hidden_dim, activation = 'relu'), 
            tf.keras.layers.Dense(units = self.hidden_dim, activation = 'relu')


        ], name = "encoder")

        self.mu_layer = tf.keras.layers.Dense(self.latent_size)
        self.logvar_layer = tf.keras.layers.Dense(self.latent_size)

        ############################################################################################
        # TODO: Implement the fully-connected decoder architecture described in the notebook.      #
        # Specifically, self.decoder should be a network that inputs a batch of latent vectors of  #
        # shape (N, Z) and outputs a tensor of estimated images of shape (N, 1, H, W).             #
        ############################################################################################
        # Replace "pass" statement with your code
        
        # decoder structure 

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation = "relu"),  

            tf.keras.layers.Dense(self.hidden_dim, activation = "relu"), 

            tf.keras.layers.Dense(self.hidden_dim, activation = "relu"), 

            tf.keras.layers.Dense(784, activation = "sigmoid"),
            tf.keras.layers.Reshape((1, 28, 28))
            
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

        ############################################################################################
        # TODO: Implement the forward pass by following these steps                                #
        # (1) Pass the input batch through the encoder model to get posterior mu and logvariance   #
        # (2) Reparametrize to compute  the latent vector z                                        #
        # (3) Pass z through the decoder to resconstruct x                                         #
        ############################################################################################
        # Replace "pass" statement with your code
        
        x_hat = self.encoder(x)
        #print(x_hat.shape)
        mu = self.mu_layer(x_hat)
        #print(mu.shape)
        logvar = self.logvar_layer(x_hat)
        #print(logvar.shape)
        latent_vector = reparametrize(mu, logvar)
        x_hat = self.decoder(latent_vector)

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

    ################################################################################################
    # TODO: Reparametrize by initializing epsilon as a normal distribution and scaling by          #
    # posterior mu and sigma to estimate z                                                         #
    ################################################################################################
    # Replace "pass" statement with your code
    
    epsilon = tf.random.normal(shape = mu.shape, mean = 0, stddev = 1) # zero mean and unit variance
    z = tf.sqrt(tf.exp(logvar)) * epsilon + mu

    ################################################################################################
    #                              END OF YOUR CODE                                                #
    ################################################################################################
    return z


def bce_function(x_hat, x):
    """
    Computes the reconstruction loss of the VAE.
    
    Inputs:
    - x_hat: Reconstructed input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    
    Returns:
    - reconstruction_loss: Tensor containing the scalar loss for the reconstruction loss term.
    """
    bce_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.SUM,
    )
    reconstruction_loss = bce_fn(x, x_hat) * x.shape[
        -1]  # Sum over all loss terms for each data point. This looks weird, but we need this to work...
    return reconstruction_loss


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).
    Returned loss is the average loss per sample in the current batch.

    Inputs:
    - x_hat: Reconstructed input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
    - logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension
    
    Returns:
    - loss: Tensor containing the scalar loss for the negative variational lowerbound
    """

    ################################################################################################
    # TODO: Compute negative variational lowerbound loss as described in the notebook              #
    ################################################################################################
    # Replace "pass" statement with your code
    
    reconstruction_loss = bce_function(x_hat, x)
    kl_div = -0.5 * tf.reduce_sum(1 + (logvar) - tf.square(mu) - tf.exp(logvar)) # don't need axis spec here 
    loss = reconstruction_loss + kl_div
    loss = tf.reduce_mean(loss)

    ################################################################################################
    #                            END OF YOUR CODE                                                  #
    ################################################################################################
    return loss # in case this doesn't work, you may have to change this so you sum over samples and divide by batch index

class CVAE(tf.keras.Model):
#     def __init__(self, input_size, num_classes=10, latent_size=15):
#         super(CVAE, self).__init__()
#         self.input_size = input_size  # H*W
#         self.latent_size = latent_size  # Z
#         self.num_classes = num_classes  # C
#         self.hidden_dim = 800  # H_d
#         self.encoder = None
#         self.mu_layer = None
#         self.logvar_layer = None
#         self.decoder = None

#         ############################################################################################
#         # TODO: Define a FC encoder as described in the notebook that transforms the image--after  #
#         # flattening and now adding our one-hot class vector (N, H*W + C)--into a hidden_dimension #               #
#         # (N, H_d) feature space, and a final two layers that project that feature space           #
#         # to posterior mu and posterior log-variance estimates of the latent space (N, Z)          #
#         ############################################################################################
#         # Replace "pass" statement with your code
        
#         self.encoder = tf.keras.Sequential([ 
#             tf.keras.layers.Flatten(),
            
#             #tf.keras.layers.Dense(input_shape = self.input_size, units = self.hidden_dim, activation = 'relu'),
#             tf.keras.layers.Dense(input_dim = (self.input_size + self.num_classes), units = self.hidden_dim, activation = 'relu'),


#             #tf.keras.layers.Dense(input_shape = self.hidden_dim, units = self.hidden_dim, activation = 'relu'),
#             tf.keras.layers.Dense(units = self.hidden_dim, activation = 'relu'),


#             #tf.keras.layers.Dense(input_shape = self.hidden_dim, units = self.hidden_dim, activation = 'relu'), 
#             tf.keras.layers.Dense(units = self.hidden_dim, activation = 'relu'), 


#         ], name = "encoder")

#         self.mu_layer = tf.keras.layers.Dense(self.latent_size)
#         self.logvar_layer = tf.keras.layers.Dense(self.latent_size)

#         ############################################################################################
#         # TODO: Define a fully-connected decoder as described in the notebook that transforms the  #
#         # latent space (N, Z + C) to the estimated images of shape (N, 1, H, W).                   #
#         ############################################################################################
#         # Replace "pass" statement with your code
        
#         self.decoder = tf.keras.Sequential([
#             tf.keras.layers.Dense(self.hidden_dim, activation = "relu"),  

#             tf.keras.layers.Dense(self.hidden_dim, activation = "relu"), 

#             tf.keras.layers.Dense(self.hidden_dim, activation = "relu"), 

#             tf.keras.layers.Dense(self.input_size, activation = "sigmoid"),
#             tf.keras.layers.Reshape((-1, 28, 28))
            
#         ], name = "decoder")

#         ############################################################################################
#         #                                      END OF YOUR CODE                                    #
#         ############################################################################################

#     def call(self, x, c):
#         """
#         Performs forward pass through FC-CVAE model by passing image through 
#         encoder, reparametrize trick, and decoder models
    
#         Inputs:
#         - x: Input data for this timestep of shape (N, 1, H, W)
#         - c: One hot vector representing the input class (0-9) (N, C)
        
#         Returns:
#         - x_hat: Reconstruced input data of shape (N, 1, H, W)
#         - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
#         - logvar: Matrix representing estimated variance in log-space (N, Z),  with Z latent space dimension
#         """

#         ############################################################################################
#         # TODO: Implement the forward pass by following these steps                                #
#         # (1) Pass the concatenation of input batch and one hot vectors through the encoder model  #
#         # to get posterior mu and logvariance                                                      #
#         # (2) Reparametrize to compute the latent vector z                                         #
#         # (3) Pass concatenation of z and one hot vectors through the decoder to resconstruct x    #
#         ############################################################################################
#         # Replace "pass" statement with your code
        
#        # reshape??
#         x = tf.reshape(x, (x.shape[0], -1))
       
#         x_hat = self.encoder(tf.concat((x, c), axis = 1))
        
#         mu = self.mu_layer(x_hat)
        
#         logvar = self.logvar_layer(x_hat)
        
#         latent_vector = reparametrize(mu, logvar)
#         x_hat = self.decoder(tf.concat((latent_vector, c), axis = 1))
       

#         ############################################################################################
#         #                                      END OF YOUR CODE                                    #
#         ############################################################################################
#         return x_hat, mu, logvar
