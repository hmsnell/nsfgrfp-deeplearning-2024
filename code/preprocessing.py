import numpy as np
from functools import reduce
import string
import pandas as pd
from sklearn.model_selection import train_test_split

def process_string(text):
  """
  Converts a string to lowercase and splits it into a list of words.

  Args:
    text: The input string.

  Returns:
    A list of words in lowercase.
  """

  # Convert the string to lowercase
  lowercase_text = text.lower()

  # Split the string into a list of words on spaces
  word_list = lowercase_text.split()

  # Return the list of words
  return word_list

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

    else:
      frequency_dict[word] = 1

  return frequency_dict
  
  # Return the dictionary of word frequencies

def strip_punct(text):
    letters = set(string.ascii_letters + " ")
    words = text.split()
    filtered_words = [word for word in words if all(char in letters for char in word)]
    return " ".join(filtered_words)

def string_to_freq(text):
    letters = set(string.ascii_letters)
    text = strip_punct(text)
    word_list = process_string(text)
    word_list = remove_single_char_elements(word_list)
    freq = word_frequency(word_list)

    freq = {key: value for key, value in freq.items() if all(char in letters for char in key)}

    
    vocab = list(freq.keys())
    vocab_size = len(freq.keys())
    indices = list(enumerate(vocab))
    vocabulary = {}
    for index,word in indices:
       vocabulary[word] = index
   
    
    return vocab, vocab_size, indices, freq, word_list, vocabulary

def concatenate_text_column(df, text_column):
    """
    Concatenates the text in a specific column of a DataFrame into a 1D NumPy array.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the text data.
        text_column (str): The name of the column with the text data.

    Returns:
        np.ndarray: A 1D NumPy array containing the concatenated text.
    """

    text_data = df[text_column].tolist()
    concatenated_text = np.array(" ".join(text_data))

    return concatenated_text

def split_csv_train_test(csv_file, text_column, train_ratio=0.8):
    """
    Splits a CSV file into train and test sets randomly based on the text column.

    Args:
        csv_file (str): Path to the CSV file.
        text_column (str): Name of the column containing the text data.
        train_ratio (float): Ratio of data to include in the training set (default 0.8).

    Returns:
        tuple: Two pandas DataFrames containing the training and testing data.
    """

    df = pd.read_csv(csv_file)
    train_df, test_df = train_test_split(df, test_size=1-train_ratio, random_state=42)

    return train_df, test_df

# Example usage:
# csv_file = "../data/pdf_texts.csv"
# text_column = "text"
# train_df, test_df = split_csv_train_test(csv_file, text_column)

# train = str(concatenate_text_column(train_df, "text"))
# test = str(concatenate_text_column(test_df, "text"))

def remove_single_char_keys(vocabulary):
  """
  Removes keys and values from a dictionary where the key is a single character,
  except for the characters "a" and "i".

  Args:
    vocabulary: The dictionary to modify.

  Returns:
    The modified dictionary.
  """

  # Iterate through the dictionary items
  for key in list(vocabulary.keys()):
    # Check if the key is a single character and not "a" or "i"
    if len(key) == 1 and key not in ("a", "i"):
      # Remove the key-value pair from the dictionary
      del vocabulary[key]

  # Return the modified dictionary
  return vocabulary

def mask_rare_words_in_strings(string1, string2, rare_words):
    """
    Masks rare words in two strings based on a list of rare words.

    Args:
        string1 (str): The first string.
        string2 (str): The second string.
        rare_words (list): A list of rare words to mask.

    Returns:
        tuple: Two lists containing the masked words from string1 and string2.
    """

    # Convert strings to lowercase and split into word lists
    word_list1 = string1.lower().split()
    word_list2 = string2.lower().split()

    # Mask rare words in each word list
    masked_list1 = ["<UNK>" if word in rare_words else word for word in word_list1]
    masked_list2 = ["<UNK>" if word in rare_words else word for word in word_list2]

    return masked_list1, masked_list2

def remove_single_char_elements(word_list):
  """
  Removes elements from a list where the element is a single character,
  except for the characters "a" and "i".

  Args:
    word_list: The list to modify.

  Returns:
    The modified list.
  """

  # Use list comprehension to create a new list with filtered elements
  filtered_list = [word for word in word_list if not (len(word) == 1 and word not in ("a", "i"))]

  # Return the filtered list
  return filtered_list

def remove_single_char_from_string(text):
  """
  Removes single-character words from a string, except for "a" and "i".

  Args:
    text: The input string.

  Returns:
    The modified string.
  """

  # Convert the string to a list of words
  word_list = text.split()


  # Filter the word list
  filtered_list = remove_single_char_elements(word_list)


  # Join the filtered list back into a string
  modified_text = " ".join(filtered_list)

  # Return the modified string
  return modified_text

def get_data(train, test):
    vocabulary = {}
    train = strip_punct(train) + "<END_train>"
    test = strip_punct(test) + "<END>"
    corpus = train + test

    corpus = remove_single_char_from_string(corpus)
    vocab, vocab_size, indices, freq, wordlist, vocabulary = string_to_freq(corpus)


    num_words_to_replace = 2
    rare_words = {word for word, count in freq.items() if count <= num_words_to_replace}

    train, test = corpus.split("<END_train>")
    test = strip_punct(test)
    train = strip_punct(train)
    train, test = mask_rare_words_in_strings(train, test, rare_words )
    test = [word for word in test if all(ord(char) < 128 for char in word)]
    train = [word for word in train if all(ord(char) < 128 for char in word)]


    for word in train+test:  
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)



    vocab_size = len(vocabulary)

    for word in train:
        if word not in vocabulary:
            print(f"Word not found in vocabulary: '{word}'")

    assert reduce(lambda x, y: x and (y in vocabulary), test)


    assert all(0 <= value < vocab_size for value in vocabulary.values()) # good 



    train_data = list(map(lambda x: vocabulary[x], train))
    test_data  = list(map(lambda x: vocabulary[x], test))


    return train_data, test_data, vocabulary

def preprocess_complete(csv_file, text_column):
    train_df, test_df = split_csv_train_test(csv_file, text_column)
    train = str(concatenate_text_column(train_df, "text"))
    test = str(concatenate_text_column(test_df, "text"))
    return get_data(train,test)

#text = "text"
#final_train, final_test, final_vocab = preprocess_complete("../data/pdf_texts.csv",text)