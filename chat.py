from utils import *
from preprocessing import *

bot_name = "Jarvis"

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

def get_q_response(txt):
    input_sequence, num_words, _, _, _ = readDataTokenize('../topic.txt', 1500)
    max_sequence_len = max([len(x) for x in input_sequence])
    x,y = preprocessing(input_sequence, num_words) # x contains all the sentences, y (one-hot coded) contains all the next words, so the training basically learns which word is best fit as the next word of the sentence.
    print(x.shape, y.shape)
    model = build_model(num_words, max_sequence_len)
    print(model.summary())
    model.fit(x, y, epochs = 150, verbose = 1)
    model.save("output/")


