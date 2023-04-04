import tensorflow as tf
from tensorflow.keras import regularizers
from preprocessing import *
from transformer import *
from utils import *

# there are 2 parts of the problem:
# 1. generate the full sequence to get the sentence 

def build_model(total_words, max_sequence_len):
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(total_words, 100, input_length = max_sequence_len-1),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150,return_sequences = True)),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.LSTM(100),
    #     tf.keras.layers.Dense(total_words//2, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)),
    #     tf.keras.layers.Dense(total_words, activation = 'softmax')
    # ])

    # transformer model
    # inputs = tf.keras.layers.Input(shape=(max_sequence_len - 1, ))
    # x = TransformerEncoder(num_layers=12, d_model=32, num_heads=12, dff=64, dropout_rate=0.1)(inputs)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(64, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # outputs = tf.keras.layers.Dense(total_words, activation='softmax')(x)
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)

    embed_dim = 32  # Embedding size for each token
    num_heads = 12  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = tf.keras.layers.Input(shape=(max_sequence_len - 1,))
    embedding_layer = TokenAndPositionEmbedding(max_sequence_len - 1, total_words, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(total_words/2, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(total_words, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['mse', 'accuracy'])
    return model

def train():
    input_sequence, num_words, _, _, _ = readDataTokenize('../questions.txt', 1500)
    max_sequence_len = max([len(x) for x in input_sequence])
    x,y = preprocessing(input_sequence, num_words) # x contains all the sentences, y (one-hot coded) contains all the next words, so the training basically learns which word is best fit as the next word of the sentence.
    print(x.shape, y.shape)
    model = build_model(num_words, max_sequence_len)
    print(model.summary())
    model.fit(x, y, epochs = 150, verbose = 1)
    model.save("output/")

def get_response(model, test_sequence):
    answers = getLine('../answers.txt', 1000)
    input_sequence, num_words, token, _, corpus= readDataTokenize('../questions.txt', 1000)
    i = 0
    got_response = False

    while len(test_sequence.split(" ")) < 30:
        if not searchF(test_sequence, corpus)[0]:
            break

        i = searchF(test_sequence, corpus)[1] 

        token_list = token.texts_to_sequences([test_sequence])[0]
        max_sequence_len = max([len(x) for x in input_sequence])
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen = max_sequence_len - 1, padding = 'pre')
        predicted = list(model.predict(token_list, verbose = 2)[0])
        predicted = predicted.index(max(predicted))

        output_words = ""
        for word, index in token.word_index.items():
            if index == predicted:
                output_words = word
                got_response = True
                break
        test_sequence += " "+ output_words

    question, answer = corpus[i], answers[i]
    print("Question:", question, "?")
    print("Answer:", answer)

    if got_response:
        return question, answer 

    return question, "I don't understand"



train()
model = tf.keras.models.load_model('../output')
s = input()
get_response(model, s)


