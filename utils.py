import re
import string
import collections
import nltk.tokenize
import os 
import json

# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.tokenize import RegexpTokenizer
# from nltk.probability import FreqDist
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk.sentiment.vader import SentimentIntensityAnalyzer

def KMPSearch(pat, txt):
    M = len(pat)
    N = len(txt)
 
    # create lps[] that will hold the longest prefix suffix
    # values for pattern
    lps = [0]*M
    j = 0 # index for pat[]
 
    # Preprocess the pattern (calculate lps[] array)
    computeLPSArray(pat, M, lps)
 
    i = 0 # index for txt[]
    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1
 
        if j == M:
            j = lps[j-1]
            return True
 
        # mismatch after j matches
        elif i < N and pat[j] != txt[i]:
            # Do not match lps[0..lps[j-1]] characters,
            # they will match anyway
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
    return False 

def computeLPSArray(pat, M, lps):
    len = 0 # length of the previous longest prefix suffix
 
    lps[0] # lps[0] is always 0
    i = 1
 
    # the loop calculates lps[i] for i = 1 to M-1
    while i < M:
        if pat[i]== pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            # This is tricky. Consider the example.
            # AAACAAAA and i = 7. The idea is similar
            # to search step.
            if len != 0:
                len = lps[len-1]
 
                # Also, note that we do not increment i here
            else:
                lps[i] = 0
                i += 1

def getLine(fileName, size=0):
    with open(fileName) as f:
        if size > 0:
            input_sequences = [remove_punctuation(x.strip()[:len(x) - 1]) for x in f.readlines() if len(x) > 10][:size]
        else:
            input_sequences = [remove_punctuation(x.strip()[:len(x) - 1]) for x in f.readlines() if len(x) > 10]

    return input_sequences


def remove_punctuation(data):
    return (re.sub(r'[^\w\s]', '', str(data)))

# def remove_fillers(data):
#     data = lower(data)
#     fillers = 


def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def searchF(txt, corpus): # pattern matching from start
    length = len(txt)
    num_match = 0
    index = 0
    for i in range(len(corpus)):
        # print(txt)
        # print(corpus[index])
        # print("================================")

        if txt.lower() == corpus[i][:length].lower():
            num_match += 1 # This is for checking if there are more than 1 text that matches with this pattern, if yes then say "I don't understand" because of ambiguity
            index = i
            

        # if KMPSearch(txt, text):
        #     return True 
        # else:
        #     return txt 

    if num_match == 1:
        return True, index
    elif num_match > 1:
        return False

    return False, index

def search(txt, corpus): # -> KMP pattern matching
    num_match = 0
    index = 0
    for i in range(len(corpus)):
        # print(txt)
        # print(corpus[i])
        # print("================================")
        if KMPSearch(txt.lower(), corpus[i].lower()) and abs(len(txt) - len(corpus[i])) <= 1:
            return True, i 

        if KMPSearch(txt.lower(), corpus[i].lower()) and len(txt) < len(corpus[i]):
            num_match += 1
            index = i

    if num_match == 1: 
        return True, index
    elif num_match > 1:
        return False

    return False, index

def getIntents(fileName):
    with open(fileName) as f:
        x = json.load(f)

    return x['intents']

def getPatterns(fileName):
    patterns = []
    with open(fileName) as f:
        x = json.load(f)

    intents = x['intents']
    for intent in intents:
        patterns += intent['patterns']

    return patterns 

def getTags(fileName):
    tags = []
    with open(fileName) as f:
        x = json.load(f)

    intents = x['intents']
    for intent in intents:
        for pattern in intent['patterns']:    
            tags.append(intent['tag'])

    return tags

def searchWiki(text, data):
    num_match = 0 
    index = 0
    for i in range(len(data)):
        if KMPSearch(text.lower(), data[i]['title'].lower()):
            num_match += 1
            index = i 
    
    if num_match == 1:
        return d[index]['text']

    return "I don't understand"

def getWiki():
    files = os.listdir("../wikidata")
    data = []
    for file in files:
        file = "../wikidata/" + file 
        with open(file) as f:
            x = json.load(f)

        data += x 

    text = input()
    print(searchWiki(text, data))
    return data 


# get the verbs
# get the question words 
# get the subject/objects 
def context_scraping(sentence):
    question_words = ['when', 'why', 'what', 'how', 'who']



