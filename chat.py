from utils import *
from preprocessing import *
import random

bot_name = "Jarvis"

def get_response_v1(model, test_sequence):
    answers = getLine('../answers.txt', 1500)
    input_sequence, num_words, token, _, corpus= readDataTokenize('../questions.txt', 1500)
    i = 0
    got_response = False

    while len(test_sequence.split(" ")) < 30:
        # run until the sentence cannot be found in the corpus
        if not searchF(test_sequence.lower(), corpus.lower())[0]:
            break

        i = searchF(test_sequence, corpus)[1] 

        token_list = token.texts_to_sequences([test_sequence])[0]
        max_sequence_len = max([len(x) for x in input_sequence])
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen = max_sequence_len - 1, padding = 'pre')
        predicted = list(model.predict(token_list, verbose = 2)[0])
        predicted = predicted.index(max(predicted))

        output_words = ""
        # add in more words that have the same index in to the resulting sentence. If this for loop never runs, then the sentence cannot be found
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

def get_response_v2(model, test_sequence):
    intents = getIntents("intents.json")
    tags = getTags("intents.json")
    input_sequence, num_words, token, _, corpus = readIntents()
    i = 0
    got_response = False

    # found the one correct pattern -> break true
    # did not find anything after 30 words -> break false
    # found more than one pattern -> break false
    # detect pattern but lengths don't match -> keep running 

    while len(test_sequence.split(" ")) < 30:
        search_result = search(test_sequence, corpus)
        if type(search_result) == tuple and search_result[0]:
            got_response = True
            i = search_result[1]
            break
        elif type(search_result) == bool:
            break

        i = search_result[1] 

        token_list = token.texts_to_sequences([test_sequence])[0]
        max_sequence_len = max([len(x) for x in input_sequence])
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen = max_sequence_len - 1, padding = 'pre')
        predicted = list(model.predict(token_list, verbose = 0)[0])
        predicted = predicted.index(max(predicted))

        output_words = ""
        for word, index in token.word_index.items():
            if index == predicted:
                output_words = word
                break
        test_sequence += " "+ output_words
        print(test_sequence)

    question, tag = corpus[i], tags[i]

    for intent in intents:
        if intent['tag'] == tag: responses = intent['responses']

    answer = responses[random.randint(0, len(responses) - 1)]

    if got_response:
        print("Question:", question)
        print("Answer:", answer)
        return question, answer 

    print("I don't understand, can you repeat that boss?")
    return question, "I don't understand, can you repeat that boss?"

def get_q_response(txt):
    input_sequence, num_words, _, _, _ = readDataTokenize('../topic.txt', 1500)
    max_sequence_len = max([len(x) for x in input_sequence])
    x,y = preprocessing(input_sequence, num_words) # x contains all the sentences, y (one-hot coded) contains all the next words, so the training basically learns which word is best fit as the next word of the sentence.
    print(x.shape, y.shape)
    model = build_model(num_words, max_sequence_len)
    print(model.summary())
    model.fit(x, y, epochs = 150, verbose = 1)
    model.save("output/")



