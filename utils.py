import re
import string
import collections
import nltk.tokenize

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# from PIL import Image

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
            print ("Found pattern at index", str(i-j))
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


def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def searchF(txt, corpus):
    length = len(txt)
    for index in range(len(corpus)):
        # print(txt)
        # print(corpus[index])
        # print("================================")

        if txt == corpus[index][:length]:
            return True, index

        # if KMPSearch(txt, text):
        #     return True 
        # else:
        #     return txt 
    return False, index

def search(txt, corpus):
    for index in range(len(corpus)):
        # print(txt)
        # print(corpus[index])
        # print("================================")

        if KMPSearch(txt, text):
            return True, index

    return False, index
