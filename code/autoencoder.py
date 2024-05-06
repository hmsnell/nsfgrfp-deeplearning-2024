# Main reference: https://github.com/Tony607/Keras_Deep_Clustering/blob/master/Keras-DEC.ipynb

import tensorflow as tf
import keras.backend as K
from keras.layers import Layer, InputSpec, Dense
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.cluster import KMeans 
from preprocessing import *
from sklearn.metrics import *
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import os
import random



class ClusteringLayer(Layer): # reference: https://github.com/Tony607/Keras_Deep_Clustering/blob/master/Keras-DEC.ipynb
    """
    Clustering layer converts input sample (feature) to soft label.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape = (None, input_dim))
        self.clusters = self.add_weight(shape = (self.n_clusters, input_dim), initializer = 'glorot_uniform', name = 'clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.        
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def encoder(input_dim, encoding_dim, vocab_size, embedding_dim): 
    inputs = tf.keras.layers.Input(shape = (input_dim,))                               # get inputs
    #embedded = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)             # embed inputs
    #embedded_flat = tf.keras.layers.Flatten()(embedded)                                 # flatten embeddings
    encoded = tf.keras.layers.Dense(encoding_dim, activation = 'relu')(inputs)          # dense layer (can add more if needed)
    encoded2 = tf.keras.layers.Dense(encoding_dim, activation = 'relu')(encoded)
    encoded3 = tf.keras.layers.Dense(encoding_dim, activation =  'relu')(encoded2)

    encoder = tf.keras.models.Model(inputs, encoded3, name = 'encoder')                  # build model 
    
    return encoder

def decoder(input_dim, encoding_dim):
    encoded_input = tf.keras.layers.Input(shape=(encoding_dim,))                        # get inputs from encoder
    decoded = tf.keras.layers.Dense(input_dim, activation = 'sigmoid')(encoded_input)   # dense layer (can add more)
    decoder = tf.keras.models.Model(encoded_input, decoded, name = 'decoder')           # build model
    return decoder

def autoencoder(encoder, decoder): 
    autoencoder = tf.keras.models.Model(encoder.input, decoder(encoder.output), name = 'autoencoder')
    return autoencoder

def kmeans_clustering(x, y, n_clusters): 
    kmeans = KMeans(n_clusters = n_clusters)
    y_pred_kmeans = kmeans.fit_predict(x)   

    accuracy = accuracy_score(y, y_pred_kmeans)
    return accuracy

def train_noclustering(train_data, max_sequence_length):
    ## train model without clustering and save weights 

    input_dim = train_data.shape[1]
    encoding_dim = 256

    encoder_layer = encoder(input_dim, encoding_dim, max_sequence_length, embedding_dim = 2)
    decoder_layer = decoder(input_dim, encoding_dim)

    model = autoencoder(encoder_layer, decoder_layer)

    opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
    model.compile(optimizer = opt, loss = 'mse')

    model.fit(train_data, train_data, epochs = 300, batch_size = 256, shuffle = True)

    model.save_weights('./ae_noclustering')

    return encoder_layer

def train_clustering(encoder_layer, train_data, train_labels):
    ## clustering 

    batch_size = 256
    n_clusters = 2
    clustering_layer = ClusteringLayer(n_clusters)(encoder_layer.output)
    clustering_model = Model(inputs = encoder_layer.input, outputs = clustering_layer)
    clustering_model.compile(optimizer = tf.keras.optimizers.Adam(0.1), loss = 'kld')

    # initialize cluster centers 
    kmeans = KMeans(n_clusters = n_clusters, n_init = 20)
    y_pred = kmeans.fit_predict(encoder_layer.predict(train_data))

    y_pred_last = np.copy(y_pred)

    clustering_model.get_layer(name = 'clustering_layer').set_weights([kmeans.cluster_centers_])

    clustering_model.save_weights('./ae_clustering')

    # optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    # total_loss = 0
    # for (batch, (train_data, train_labels)) in enumerate(tqdm(len(train_data))):
    #     with tf.GradientTape() as tape:
    #         q = clustering_model.predict(train_data, verbose = 0)
    #         p = target_distn(q)
    #         y_pred = q.argmax(1)
    #         loss = tf.keras.losses.SparseCategoricalCrossEntropy(train_labels, y_pred)
    #         #print(loss)
    #     gradients = tape.gradient(loss, clustering_model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, clustering_model.trainable_variables))
    #     total_loss += (loss.numpy() / (batch_size))
    # return total_loss

    # deep clustering 
    
    loss = 0
    index = 0
    maxiter = 8000
    update_interval = 140
    index_array = np.arange(train_data.shape[0])
    tol = -0.0001
    

    ## train with clustering 
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q = clustering_model.predict(train_data, verbose=0)
            p = target_distn(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if train_labels is not None:
                acc = np.round(accuracy_score(train_labels, y_pred), 5)
                loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f' % (ite, acc), ' ; loss=', loss)

            # check stop criterion - model convergence
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 1 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * batch_size: min((index+1) * batch_size, train_data.shape[0])]
        loss = clustering_model.train_on_batch(x = train_data[idx], y = p[idx])
        index = index + 1 if (index + 1) * batch_size <= train_data.shape[0] else 0

    return clustering_model, loss

def test_clustering(clustering_model, loss, test_data, test_labels):
    # evaluation testing

    clustering_model.load_weights('./ae_clustering')
    # Eval.
    q = clustering_model.predict(test_data, verbose=0)
    p = target_distn(q)  # update the auxiliary target distribution p

    # evaluate the clustering performance
    y_pred = q.argmax(1)
    if test_labels is not None:
        acc = np.round(accuracy_score(test_labels, y_pred), 5)
        loss = np.round(loss, 5)
    print('Test accuracy = %.5f' % (acc), ' ; loss = ', loss)

    return p

def target_distn(q): 
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

def get_final_results(y, y_pred): 
    #confusion_matrix_out = confusion_matrix(y, y_pred)

    # plt.figure(figsize = (16, 14))
    # sns.heatmap(confusion_matrix, annot = True, fmt = "d", annot_kws = {"size": 20});
    # plt.title("Confusion matrix", fontsize = 30)
    # plt.ylabel('True label', fontsize = 25)
    # plt.xlabel('Clustering label', fontsize = 25)
    # plt.savefig('confusion_matrix.pdf')

    y_true = y.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # Confusion matrix.
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(-w)

    sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    return w

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # tf.experimental.numpy.random.seed(seed)
    # tf.set_random_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main():

    set_seed()
    sentences , labels = preprocess_complete_ver2("../data/pdf_texts.tsv", "text")

    ##### dummy data 
    
    texts = sentences
    labels = labels

    # tokenize
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    # text to sequences
    sequences = tokenizer.texts_to_sequences(texts)

    # pad sequences
    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen = max_sequence_length, padding = 'post')

    # split train and test data 
    total_sentences = len(padded_sequences)
    train_ratio = int(total_sentences * 0.8)
    #test_ratio = 1 - train_ratio

    train_data = padded_sequences[0:train_ratio]
    test_data = padded_sequences[train_ratio:total_sentences]
    train_labels = np.array(labels[0:train_ratio])
    test_labels = np.array(labels[train_ratio:total_sentences])

    #print(test_data)
    #print(test_labels)

    ######

    kmeans_accuracy = kmeans_clustering(train_data, train_labels, 2)
    print(kmeans_accuracy)

    encoder_layer = train_noclustering(train_data, max_sequence_length)
    #for epoch_id in range(300):
    #    total_loss = train_clustering(encoder_layer, train_data, train_labels)
    #    print(f"Train Epoch: {epoch_id} \tLoss: {total_loss / len(train_data):.6f}")
    clustering_model, loss = train_clustering(encoder_layer, train_data, train_labels)
    y_pred = test_clustering(clustering_model, loss, test_data, test_labels)
    #print(y_pred)
    #print(get_final_results(test_labels, y_pred))

    
if __name__ == "__main__":
     main()