import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square
from preprocessing import * 
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
import os
import argparse
from autoencoder import *

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--latent_size", type=int, default=15)
    parser.add_argument("--input_size", type=int, default=28 * 28)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()
    return args

def train_ae(model, train_loader, args):
    optimizer = tf.keras.optimizers.Adam(learning_rate = args.learning_rate)
    total_loss = 0
    for (batch, (x)) in enumerate(tqdm(train_loader)):
        with tf.GradientTape() as tape:
            x_hat = model(x)
            loss = bce_function(x_hat, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += (loss.numpy() / (args.batch_size))
    return total_loss

def save_model_weights(model, args):
    model_flag = "ae"
    output_dir = os.path.join("model_ckpts", model_flag)
    output_path = os.path.join(output_dir, model_flag)
    os.makedirs("model_ckpts", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    model.save_weights(output_path)

def load_weights(model):
    #num_classes = 10
    inputs = tf.zeros([1, 1, 28, 28])  # random data sample, change to dims of inputs 
    #labels = tf.constant([[0]])
    weights_path = os.path.join("model_ckpts", "ae", "ae")
    _ = model(inputs)
    model.load_weights(weights_path)
    return model

def main(args):
    # preprocess data
    train_data, test_data, vocab = preprocess_complete("../data/pdf_texts.csv", "text")

    print(train_data.shape())
    # get an instance of AE
    model = AE(len(vocab))

    # Load trained weights
    # if args.load_weights:
    #    model = load_weights(model)

    # pretrain autoencoder
    for epoch_id in range(args.num_epochs):
        total_loss = train_ae(model, train_data)
        print(f"Train Epoch: {epoch_id} \tLoss: {total_loss / len(train_data):.6f}")

    save_model_weights(model)


if __name__ == "__main__":
    args = parseArguments()
    main(args)

    #def loss_function(x_hat, x, mu, logvar):
#    # using sparse categorical cross-entropy loss, this is for multiclass classification where classes are mutually exclusive (soft clustering labels are only 1 per sample)
#    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False) 
#    return loss 

# def get_clustering_model(vocab):

#     #def perplexity(y_true, y_pred): 
#     #    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
#     #    perplexity = tf.exp(tf.reduce_mean(ce))
#     #    return perplexity

#     # define model, loss, accuracy
#     model = AE(len(vocab))
#     # loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False) 
#     # acc_metric  = perplexity

#     # compile model 
#     model.compile(
#         optimizer= 'adam', 
#         loss = 'mse'
#     )

#     return SimpleNamespace(
#         model = model,
#         epochs = 1,
#         batch_size = 100,
#     )

# def main():
    
#     # perform preprocessing 
#     train_data, test_data, vocab = preprocess_complete("../data/pdf_texts.csv", "text")

#     # reformat inputs (we did this for RNN so I left it here, may not need)
#     def prepare_inputs(data, window_size): 
#         data_array = np.array(data)
#         remainder = (len(data_array) - 1)%window_size
#         data_array = data_array[:-remainder]

#         X = data_array[:-1].reshape(-1, 4)
#         Y = data_array[1:].reshape(-1, 4)

#         return X, Y

#     X0, Y0  = prepare_inputs(train_data, 20)
#     X1, Y1  = prepare_inputs(test_data, 20)

#     # call model 
#     args = get_clustering_model(vocab)

#     args.model.fit(
#         X0, Y0,
#         epochs=args.epochs, 
#         batch_size=args.batch_size,
#         validation_data=(X1, Y1)
#     )

# if __name__ == '__main__':
#     main()
