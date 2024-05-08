import tensorflow as tf 
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.cluster import OPTICS
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import re
import csv
import argparse
import math
import os
import random
from tensorflow.math import sigmoid
from tqdm import tqdm
from vae import VAE, reparametrize,  kil_adavae_loss
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import utils
from scipy.optimize import minimize
import sklearn

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_cvae", action="store_true")
    parser.add_argument("--load_weights", action="store_true")
    parser.add_argument("--batch_size", type=int, default=202)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--latent_size", type=int, default=15)
    parser.add_argument("--input_size", type=int, default=28 * 28)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()
    return args

def get_data(essay_filepath, labels_filepath): 
    """
    Read in and process essay data

    Inputs:
    - essay_filepath: path to csv where essays are stored in one line 
    - labels_filepath:  path to tsv where essays are stored in one line 

    Returns:
    - essays: list of essays
    - words: all unqiue words across all essays
    - labels: labels indicating honorable mentions and winners
    """
    
    # initialize empty lists to store associated data
    essays = []
    words = []

    # open CSV and read the associated data 
    with open(essay_filepath, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
                essay = lines[1]
                essay = re.sub(r'[^\w]+(\s+|$)',' ',essay, re.UNICODE)
                words.extend(essay.split()) 
                essays.append(essay) 

    # open tsv and get labels from the associated data 
    labels = pd.read_csv(labels_filepath, sep='\t')
    labels = np.array(labels['Success']) 
    labels = np.where(labels == "Winner!",1,0)
    
    return essays, words, labels


def tokenize_essays(essay_list, word_list, max_em_length, embed_dim):
    """
    Tokenize all words for each essay

    Inputs:
    - essays: list of essays
    - words: all unqiue words across all essays
    - labels: labels indicating honorable mentions and winners

    Returns:
    - embedded_data: list of essays where every word is tokenized for each essay 

    """
    # find vocab size and max length of embedding words
    vocab_size = len(set(word_list))
    max_length = max_em_length
    # delete first entry as its irrelevant
    del essay_list[0]

    # tokenize the words
    tokenized = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenized.fit_on_texts(essay_list)
    sequences = tokenized.texts_to_sequences(essay_list)

    # choose embed dim
    embedding_dim = embed_dim
    # Pad sequences 
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

    # Create Embedding layer & pass padded seq
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    embedded_data = embedding_layer(padded_sequences)

    return embedded_data

def mask_inputs(training_labels, perc_unmasked):
    """
    Masks some of the inputs for the training labels 

    Inputs:
    - training_labels: labels for training data 
    - perc_unmasked: how much of the data to leave unmasked

    Returns:
    - is_winner_masked: masked training labels

    """
    # set seed
    np.random.seed(2470)

    # find where all training labels are 1 (meaning winners)
    where_one = np.where(training_labels == 1)[0]

    # find number of indexes corresponding to percentages
    perc_winners = max(1, int(perc_unmasked * len(where_one)))
    # randomize which indexes to unmask
    win_index = np.random.choice(where_one, perc_winners, replace=False)

    # make all labels 0
    is_winner_masked = np.zeros_like(training_labels)
    # unmask the chosen 
    is_winner_masked[win_index] = 1

    return is_winner_masked

def train_vae(model, train_loader, args, winner_index):
    """
    Train your VAE with one epoch.

    Inputs:
    - model: Your VAE instance.
    - train_loader: A tf.data.Dataset of MNIST dataset.
    - args: All arguments.
    - is_cvae: A boolean flag for Conditional-VAE. If your model is a Conditional-VAE,
    set is_cvae=True. If it's a Vanilla-VAE, set is_cvae=False.

    Returns:
    - total_loss: Sum of loss values of all batches.
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    total_loss = 0

    for (batch, (essays, labels, winner_index)) in enumerate(tqdm(train_loader)):
        with tf.GradientTape() as tape:
            x_hat, mu, logvar = model(essays)
            loss = kil_adavae_loss(x_hat, essays, mu, 1.2, logvar, winner_index)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += (loss.numpy() / args.batch_size) 
    return total_loss, mu


def save_model_weights(model, args):
    """
    Save trained VAE model weights to model_ckpts/

    Inputs:
    - model: Trained VAE model.
    - args: All arguments.
    """
    model_flag = "cvae" if args.is_cvae else "vae"
    output_dir = os.path.join("model_ckpts", model_flag)
    output_path = os.path.join(output_dir, model_flag)
    os.makedirs("model_ckpts", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    model.save_weights(output_path)
    
nu = 0.002
# nu = 0.04 # for usps
K  = 2

def relu(x):
    y = x
    y[y < 0] = 0
    return y

def dRelu(x):
    y = x
    y[x <= 0] = 0
    y[x > 0]  = np.ones((len(x[x > 0]),))
    return y

def nnScore(X, w, V, g):
    return np.dot(g(np.dot(X,V)),w)

def ocnn_obj(theta, X, nu, D, K, g, dG):
    
    w = theta[:K]
    V = theta[K:K+K*D].reshape((D, K))
    r = theta[K+K*D:]
    
    term1 = 0.5  * np.sum(w**2)
    term2 = 0.5  * np.sum(V**2)
    term3 = 1/nu * np.mean(relu(r - nnScore(X, w, V, g)))
    term4 = -r
    
    return term1 + term2 + term3 + term4

def ocnn_grad(theta, X, nu, D, K, g, dG):
    
    N = X.shape[0]
    w = theta[:K]
    V = theta[K:K+K*D].reshape((D, K))
    r = theta[K+K*D:]
    
    deriv = dRelu(r - nnScore(X, w, V, g))  
    

    term1 = np.concatenate(( w,
                             np.zeros((V.size,)),
                             np.zeros((1,)) ))

    term2 = np.concatenate(( np.zeros((w.size,)),
                             V.flatten(),
                             np.zeros((1,)) ))

    term3 = np.concatenate(( 1/nu * np.mean(deriv[:,np.newaxis] * (-g(np.dot(X, V))), axis = 0),
                             1/nu * np.mean((deriv[:,np.newaxis] * (dG(np.dot(X, V)) * -w)).reshape((N, 1, K)) * X.reshape((N, D, 1)), axis = 0).flatten(),
                             1/nu * np.array([ np.mean(deriv) ]) ))
    
    term4 = np.concatenate(( np.zeros((w.size,)),
                             np.zeros((V.size,)),
                             -1 * np.ones((1,)) ))
    
    return term1 + term2 + term3 + term4



def One_Class_NN_explicit_linear(data_train,data_test):


    X  = data_train
    D  = X.shape[1]

    g  = lambda x : x
    dG = lambda x : np.ones(x.shape)

    np.random.seed(42)
    theta0 = np.random.normal(0, 1, K + K*D + 1)

    from scipy.optimize import check_grad
    print('Gradient error: %s' % check_grad(ocnn_obj, ocnn_grad, theta0, X, nu, D, K, g, dG))

    res = minimize(ocnn_obj, theta0, method = 'L-BFGS-B', jac = ocnn_grad, args = (X, nu, D, K, g, dG),
                   options = {'gtol': 1e-8, 'disp': True, 'maxiter' : 50000, 'maxfun' : 10000})

    thetaStar = res.x

    wStar = thetaStar[:K]
    VStar = thetaStar[K:K+K*D].reshape((D, K))
    rStar = thetaStar[K+K*D:]

    pos_decisionScore = nnScore(data_train, wStar, VStar, g) - rStar
    neg_decisionScore = nnScore(data_test, wStar, VStar, g) - rStar

    print("pos_decisionScore", np.sort(pos_decisionScore))
    print("neg_decisionScore", np.sort(neg_decisionScore))


    return [pos_decisionScore,neg_decisionScore]


def One_Class_NN_explicit_sigmoid(data_train, label_train, data_test, label_test):

    X  = data_train
    D  = X.shape[1]


    g   = lambda x : 1/(1 + np.exp(-x))
    dG  = lambda x : 1/(1 + np.exp(-x)) * 1/(1 + np.exp(+x))

    np.random.seed(42)
    theta0 = np.random.normal(0, 1, K + K*D + 1)

    from scipy.optimize import check_grad
    print('Gradient error: %s' % check_grad(ocnn_obj, ocnn_grad, theta0, X, nu, D, K, g, dG))

    res = minimize(ocnn_obj, theta0, method = 'L-BFGS-B', jac = ocnn_grad, args = (X, nu, D, K, g, dG),
                   options = {'gtol': 1e-8, 'disp': True, 'maxiter' : 50000, 'maxfun' : 10000})

    thetaStar = res.x

    wStar = thetaStar[:K]
    VStar = thetaStar[K:K+K*D].reshape((D, K))
    rStar = thetaStar[K+K*D:]

    pos_decisionScore = nnScore(data_train, wStar, VStar, g) - rStar
    neg_decisionScore = nnScore(data_test, wStar, VStar, g) - rStar
    
    #print("pos_decisionScore", np.sort(pos_decisionScore))
    #print("neg_decisionScore", np.sort(neg_decisionScore))

    return [pos_decisionScore, label_train, neg_decisionScore, label_test]

def plot_scores(scores, labels, title):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(scores, labels, c=labels, cmap='coolwarm', edgecolors='k')
    plt.xlabel("Decision Scores")
    plt.ylabel("Labels")
    plt.title(title)
    legend1 = plt.legend(*scatter.legend_elements(), title="Labels", loc='upper right')
    plt.gca().add_artist(legend1)
    plt.grid(True)
    plt.savefig(title+'.png', dpi=300, bbox_inches='tight')


def main(args):

    # get the list of essays, all unique words, and the associated labels (winner vs HM)
    essays, words, labels = get_data('../../data/pdf_texts.csv', '../../data/pdf_texts.tsv')

    # tokenize the essays (word by word)
    embedded_data = tokenize_essays(essays, words, 40, 50)

    # initialize the model
    model = VAE(args.input_size, latent_size=args.latent_size)

    # split the datset into training and testing 
    X_train, X_test, y_train, y_test = train_test_split(embedded_data.numpy(), labels, test_size=0.20, random_state=42)

    # mask some of the labels for the training data
    is_winner_train = mask_inputs(y_train, 0.65)

    # shuffle and batch the data 
    train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(X_train), y_train, is_winner_train))
    train_dataset = train_dataset.shuffle(buffer_size=20).batch(X_train.shape[0])
    
    auc_train_l = []
    auc_test_l = []
    
    for i in range(10):

        # Train VAE
        for epoch_id in range(args.num_epochs):
            total_loss = train_vae(model, train_dataset, args, is_winner_train)[0]
            print(f"Train Epoch: {epoch_id} \tLoss: {total_loss / len(train_dataset):.6f}")
    
        # save the latent representation for train
        latent_rep_train = train_vae(model, train_dataset, args, is_winner_train)[1]
        
        # get the corresponding label for the train after shuffling
        y_train_dataset = train_dataset.map(lambda x, y, z: y)
        y_train_values= [y.numpy() for y in y_train_dataset]
        y_train_df = pd.DataFrame(y_train_values).transpose()
        
        # get latent representation for train too
        is_winner_test = mask_inputs(y_test, 0.0)
        test_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(X_test), y_test, is_winner_test))
        test_dataset = test_dataset.shuffle(buffer_size=20).batch(X_test.shape[0])
        latent_rep_test = train_vae(model, test_dataset, args, is_winner_test)[1]
        
        # get the corresponding label for the test after shuffling
        y_test_dataset = test_dataset.map(lambda x, y, z: y)
        y_test_values= [y.numpy() for y in y_test_dataset]
        y_test_df = pd.DataFrame(y_test_values).transpose()
    
        # convert latent representation to numpy
        latent_rep_train = latent_rep_train.numpy()
        latent_rep_test = latent_rep_test.numpy()
        
        # one-class outlier detection for both train and test latent space
        train_decisionScore, train_labels, test_decisionScore, test_labels = One_Class_NN_explicit_sigmoid(latent_rep_train, y_train_df, latent_rep_test, y_test_df)
    
        
        # Plot training data of decision score vs labels
        plot_scores(train_decisionScore, train_labels, "Training_Decision_Scores_vs_Labels2")
        
        # Plot test data of decision score vs labels
        plot_scores(test_decisionScore, test_labels, "Testing_Decision_Scores_vs_Labels2")
        
        
        auc_train = sklearn.metrics.roc_auc_score(train_labels, train_decisionScore)
        auc_test = sklearn.metrics.roc_auc_score(test_labels, test_decisionScore)
        print("Round "+str(i) + ": auc score for training dataset = " + str(auc_train))
        print("Round "+str(i) + "auc score for testing dataset = " + str(auc_test))
        auc_train_l.append(auc_train)
        auc_test_l.append(auc_test)


    auc_train_l = np.array(auc_train_l)
    print('average auc for training dataset =' + str(auc_train_l.mean()))
    print('std for training dataset =' + str(auc_train_l.std()))
    auc_test_l = np.array(auc_test_l)
    print('average auc for testing dataset =' + str(auc_test_l.mean()))
    print('std for testing dataset =' + str(auc_test_l.std()))

    ### the following large chunck of code are used to cluster the data based on its latent space and then plot them with a tsne
    # clustering = OPTICS(min_samples=5).fit(latent_rep)
    # print(clustering.labels_)

    # y_train_dataset = train_dataset.map(lambda x, y, z: y)
    # y_train_values= [y.numpy() for y in y_train_dataset]
    # y_df = pd.DataFrame(y_train_values).transpose()

    # tsne = TSNE(n_components = 2)
    # data_trans = tsne.fit_transform(latent_rep)
    
    # #print(data_trans[:, 0].shape)
    # #print(y_df.iloc[:, 0].shape)
    # plot_data = pd.DataFrame({
    # "Component 1": data_trans[:, 0],
    # "Component 2": data_trans[:, 1],
    # "Cluster": clustering.labels_.astype(int),
    # "True Label": y_df.iloc[:, 0]
    # })
    
    # # Plot using Plotly Express
    # fig = px.scatter(
    #     plot_data,
    #     x="Component 1",
    #     y="Component 2",
    #     color="Cluster",
    #     symbol="True Label",
    #     title="t-SNE Plot of Latent Representations",
    #     labels={"Component 1": "Component 1", "Component 2": "Component 2"}
    # )
    
    # fig.update_layout(
    #     legend_title="Cluster and True Label",
    #     legend=dict(
    #         traceorder="normal"
    #     )
    # )

    # fig.write_image('latent_rep_plotly_plot2.png', scale=2)


if __name__ == "__main__":
    args = parseArguments()
    main(args)
