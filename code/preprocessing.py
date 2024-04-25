#import tensorflow as tf
import numpy as np
from functools import reduce

def word_frequency(word_list):
    """
    Calculates the frequency of each unique word in a list.

    Args:
    word_list: A list of words.

    Returns:
    A dictionary where keys are unique words and values are their frequencies.
    """

    # Initialize an empty dictionary to store word frequencies
    frequency_dict = {}

    # Iterate through the word list
    for word in word_list:
        # If the word is already in the dictionary, increment its count
        if word in frequency_dict:
            frequency_dict[word] += 1

    # Otherwise, add the word to the dictionary with a count of 1
    else:
        frequency_dict[word] = 1

    # Return the dictionary of word frequencies
    return frequency_dict

def strip_punct(text):
    text = text.replace(","," ").replace("."," ").replace("("," ").replace(")"," ")
    text = text.replace("1"," ").replace("2"," ").replace("3"," ").replace("4"," ")
    text = text.replace("5"," ").replace("6"," ").replace("7"," ").replace("8"," ")
    text = text.replace("9"," ").replace("0"," ").replace("\'"," ").replace("!"," ")
    text = text.replace("-"," ").replace("_"," ").replace("?"," ").replace("["," ")
    text = text.replace("/"," ").replace("{"," ").replace("}"," ").replace("]"," ")
    return text

def replace_rare_words(frequency_dict):
  """
  Replaces the least frequent words in a frequency dictionary with "<UNK>".

  Args:
    frequency_dict: A dictionary where keys are words and values are their frequencies.

  Returns:
    A modified dictionary with rare words replaced by "<UNK>".
  """

  # Calculate the number of words to replace (10% of the total)
  num_words_to_replace = int(len(frequency_dict) * 0.1)

  # Get the least frequent words and their frequencies
  rare_words = sorted(frequency_dict.items(), key=lambda item: item[1])[:num_words_to_replace]

  # Remove rare words from the dictionary
  #for word, _ in rare_words:
  #  del frequency_dict[word]

  # Add the "<UNK>" entry with the combined frequency of rare words
  #frequency_dict["<UNK>"] = sum(count for _, count in rare_words)

  # Return the modified dictionary
  return rare_words


def get_data(train_file, test_file):
    vocabulary, vocab_size, train_data, test_data = {}, 0, [], []

    # parse training and testing files
    training = open(train_file, 'r')
    training_lines = training.readlines() # maybe read in differently? 
    print(training_lines)   

    testing = open(test_file, 'r')
    testing_lines = testing.readlines()

    # make vocabulary dictionary 
    for line in training_lines + testing_lines: 
        line_list = strip_punct(line)
        line_list = line.lower().split()
        print(line_list)
        print()
        line_unique_words = sorted(set(line_list))

        for word in line_unique_words: 
            if word not in vocabulary: 
                vocabulary[word] = vocab_size
                vocab_size = len(vocabulary)
        
        #frequency_dict[word] = word_frequency(line_list)
        #print(frequency_dict)
    
    # make training data
    for line in training_lines: 
        line_list = line.lower().split() 
        for l in line_list:    
            train_data.append(l)

    # make testing data
    for line in testing_lines: 
        line_list = line.lower().split()
        for l in line_list:    
            test_data.append(l)

    # sanity Check, make sure there are no new words in the test data.
    assert reduce(lambda x, y: x and (y in vocabulary), test_data)
    
    # sanity check, make sure that all values are withi vocab size
    assert all(0 <= value < vocab_size for value in vocabulary.values()) # good 

    # vectorize
    train_data = list(map(lambda x: vocabulary[x], train_data))
    test_data  = list(map(lambda x: vocabulary[x], test_data))

    # print("train_data", train_data)
    #return train_data, test_data, vocabulary

    # make training and testing into text files
    with open("training_look.txt", "w") as output: 
        output.write(str(train_data))

    with open("testing_look.txt", "w") as output: 
        output.write(str(test_data))
    
    #return train_data, test_data, vocabulary
        

get_data('../data/training.txt', '../data/testing.txt', )