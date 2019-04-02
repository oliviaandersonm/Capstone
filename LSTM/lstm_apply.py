from keras.models import load_model
import numpy as np
import re, os, sys
import pandas as pd
np.random.seed(7)

#OPTIONS
TEXT_FILE = 'article.txt' #make string to use
STRINGS = [] #fill to use on any number of strings
MODEL = 'nosw'

MODEL_FILE = 'short_Gw2v_LSTM.h5' #keep stopwords
TOKENIZER_FILE = 'short_lstm_gw2v_tokenizer.pickle'
MAX_LENGTH = 74


model = load_model(MODEL_FILE)

#for formatting full article -- returns list of sentences
def clean_article(text):
    text = text.replace("\n", ".")
    text = re.sub('(Mr.|Dr.|Mrs.|Jr.|Ms.)', ' ', text)
    text = text.replace('Rep.', 'representative')
    text = re.sub(r" [A-Z]\.", ' ', text)
    text = text.lower()
    #remove contractions
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    cleaned_sentences = text.split('.')
    return cleaned_sentences
#for formatting lines -- input is a string (sentence)
def format(string):
    #line = re.sub('[^A-Za-z]', ' ', string)
    string = string.replace(",", " ").strip()
    string = re.sub("[!@#$%+*:()'-]", ' ', string)
    string = ''.join([w for w in string if not w.isdigit()])
    return string


article_sentences = []

if TEXT_FILE is not None:
    with open(TEXT_FILE) as article:
        contents = article.read()
        contents = clean_article(contents)

    line_count = 0
    for line in contents:
        line_count = line_count + 1
        line_formatted = format(line)
        #line_inquotes = "\'" + line_formatted + "\'"
        if len(line_formatted) > 2:
            article_sentences.append(line_formatted)

if len(STRINGS) != 0:
    article_sentences = STRINGS

#load tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
with open(TOKENIZER_FILE, 'rb') as f:
    tokenizer = pickle.load(f)

X = tokenizer.texts_to_sequences(article_sentences)
X = pad_sequences(X, maxlen=MAX_LENGTH)

prediction = model.predict_classes(X)

for i in range(len(X)):
    print('%s\nLabel:%d' % (article_sentences[i], prediction[i]))
