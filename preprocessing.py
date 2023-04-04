import tensorflow as tf
import numpy as np
import keras.utils as ku
import re 
from utils import *


def preprocessing(input_sequences, total_words):
    # pad the rest of the word vector with 0's

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    # basically label is the probability of the a word given the current history/sequence, the lower the number the lower the probability
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label

# predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)


tokenizer = tf.keras.preprocessing.text.Tokenizer()

def get_sequence_of_tokens(corpus):
    labels = []
    ## tokenization, the "tokenizer" represents the vocabulary taken from the corpus, basically all the words in the corpus
    # change the word into value corresponding to its frequency, the lower the more frequent, the higher the less, the dictionary "word_index" maps each word with such values
    tokenizer.fit_on_texts(corpus) 
    total_words = len(tokenizer.word_index) + 1
    
    ## convert sequence to sequence of tokens 
    input_sequences = []
    for line in corpus:
        # convert the sequences into vectors with each element being the values of dictionary "word_index"
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            print(token_list[i])
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
            labels.append(corpus.index(line))


    return input_sequences, total_words, tokenizer, np.array(labels), corpus

def get_n_gram(corpus):
    labels = []
    ## tokenization, the "tokenizer" represents the vocabulary taken from the corpus, basically all the words in the corpus
    # change the word into value corresponding to its frequency, the lower the more frequent, the higher the less, the dictionary "word_index" maps each word with such values
    tokenizer.fit_on_texts(corpus) 
    total_words = len(tokenizer.word_index) + 1
    
    ## convert sequence to sequence of tokens 
    input_sequences = []
    for line in corpus:
        # convert the sequences into vectors with each element being the values of dictionary "word_index"
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
            labels.append(corpus.index(line))


    return input_sequences, total_words, tokenizer, np.array(labels), corpus


def readDataTokenize(fileName, size=0):
    with open(fileName) as f:
        if size > 0:
            input_sequences = [remove_punctuation(x.strip()[:len(x) - 1]) for x in f.readlines() if len(x) > 10][:size]
        else:
            input_sequences = [remove_punctuation(x.strip()[:len(x) - 1]) for x in f.readlines() if len(x) > 10]

    inp_sequences, total_words, tokenizer, labels, corpus = get_sequence_of_tokens(input_sequences) 

    return inp_sequences, total_words, tokenizer, labels,  corpus


