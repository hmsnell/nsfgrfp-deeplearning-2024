import tensorflow as tf 
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
            loss = kil_adavae_loss(x_hat, essays, mu, 1.0, logvar, winner_index)
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


def main(args):

    # get the list of essays, all unique words, and the associated labels (winner vs HM)
    essays, words, labels = get_data('data/pdf_texts.csv', 'data/pdf_texts.tsv')

    # tokenize the essays (word by word)
    embedded_data = tokenize_essays(essays, words, 40, 50)

    # initialize the model
    model = VAE(args.input_size, latent_size=args.latent_size)

    # split the datset into training and testing 
    split_ds = X_train, X_test, y_train, y_test = train_test_split(embedded_data.numpy(), labels, test_size=0.20, random_state=42)

    # mask some of the labels for the training data
    is_winner = mask_inputs(y_train, 0.65)

    # shuffle and batch the data 
    train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(X_train), y_train, is_winner))
    train_dataset = train_dataset.shuffle(buffer_size=20).batch(X_train.shape[0])

    # Train VAE
    for epoch_id in range(args.num_epochs):
        total_loss = train_vae(model, train_dataset, args, is_winner)[0]
        print(f"Train Epoch: {epoch_id} \tLoss: {total_loss / len(train_dataset):.6f}")

    # save the latent representation 
    latent_rep = train_vae(model, train_dataset, args, is_winner)[1]
 
    # save the model
    save_model_weights(model, args)


if __name__ == "__main__":
    args = parseArguments()
    main(args)
