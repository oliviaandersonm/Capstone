from sklearn.externals import joblib
import numpy as np
import pandas as pd
import re
import sys, os
import csv

MODEL = 'nosw'

if MODEL == 'rmsw':
    MODEL_FILE = os.path.expanduser('~/Capstone/LSTM/Gw2v_LSTM.h5')
    TOKENIZER_FILE = os.path.expanduser('~/Capstone/LSTM/lstm_gw2v_tokenizer.pickle')
    MAX_LENGTH = 45
    AVG_FILE = 'averages_lstm_rmsw_2.csv'
elif MODEL == 'short':
    MODEL_FILE = os.path.expanduser('~/Capstone/LSTM/short_Gw2v_LSTM.h5')
    TOKENIZER_FILE = os.path.expanduser('~/Capstone/LSTM/short_lstm_gw2v_tokenizer.pickle')
    MAX_LENGTH = 74
    AVG_FILE = 'averages_lstm_short_2.csv'
else: #keep stopwords
    MODEL_FILE = os.path.expanduser('~/Capstone/LSTM/Gw2v_nosw_LSTM.h5')
    TOKENIZER_FILE = os.path.expanduser('~/Capstone/LSTM/lstm_gw2v_nosw_tokenizer.pickle')
    MAX_LENGTH = 74
    AVG_FILE = 'author_averages_nosw.csv'
#load tokenizer & model
from keras.models import load_model
model = load_model(MODEL_FILE)
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
with open(TOKENIZER_FILE, 'rb') as f:
    tokenizer = pickle.load(f)

def to_csv(s):
    download_dir = "x.csv"

    csv = open(download_dir, "w")

    columnTitleRow = "sentence\n"
    csv.write(columnTitleRow)

    for sentence in s:
        sentence = format(sentence)
        row = sentence + "\n"
        csv.write(row)


    csv.close()


def apply_lstm(csv, length):

    dataset = pd.read_csv(csv, quotechar=None, quoting=3)

    sentences = []
    for i in range(dataset.shape[0]):
        sentences.append(dataset.iloc[i][0])

    sentences = [str(w) for w in sentences]

    X = tokenizer.texts_to_sequences(sentences)
    X = pad_sequences(X, maxlen=MAX_LENGTH)

    try:
        prediction = model.predict_classes(X)
    except:
        prediction = [1,0,0,0,1]


    num_subj = 0
    num_obj = 0
    for s in range(length):
        try:
            if prediction[s] == 1:
                num_subj += 1
            else:
                num_obj += 1
        except(IndexError):
            continue

    ratio_obj = num_obj / (num_obj + num_subj)
    return ratio_obj

#for formatting full article -- returns list of sentences
def clean_article(text):
    text = str(text)
    text = text.replace("\n", ".")
    text = re.sub('(Mr.|Dr.|Mrs.|Jr.|Ms.)', ' ', text)
    text = text.replace('Rep.', 'representative')
    text = re.sub(r" [A-Z]\.", ' ', text)
    text = ''.join([w for w in text if not w.isdigit()])
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
    return string

def get_avg(source):
    sentences = []
    ratios = []
    print("Opening Source " + source)
    file = open(os.path.expanduser('~/Capstone/data/all-the-news/%s.csv') % source)
    file.readline()
    line = file.readline()
    while(line):
        sentences = clean_article(line)
        to_csv(sentences)
        ratio = apply_lstm('x.csv', len(sentences))
        ratios.append(ratio)
        line = file.readline()

    file.close()
    print("Closed " + source)
    total = 0
    for ratio in ratios:
        total += ratio
    avg = total / len(ratios)

    return avg

############
# sources = ['NewYorkTimes', 'Breitbart', 'CNN', 'BusinessInsider', 'Atlantic',
#  'FoxNews', 'TalkingPointsMemo', 'BuzzfeedNews', 'NationalReview', 'NewYorkPost',
#  'Guardian', 'NPR', 'Reuters', 'Vox', 'WashingtonPost']
# sources = ['abc-news', 'associated-press', 'bbc-news', 'bloomberg', 'cnbc',
#         'daily-mail', 'financial-times', 'fortune',
#         'msnbc', 'nbc-news', 'politico',
#         'the-wall-street-journal',
#         'time', 'vice-news', 'usa-today']
#sources = ['cnn_op', 'nyt_op', 'atlantic_op', 'fox_op', 'guardian_op', 'wp_op', 'usa_op']
sources = ['Rob Stein', 'Sarah McCammon','Suleiman Al-Khalidi', 
     'Sam Forgione', 'Peter Beinart', 'Ana Swanson', 'Zoe Tillman',
     'Howard Kurtz', 'David Morgan', 'David Nakamura', 'Lisa Lambert',
     'Dino Grandoni', 'Laura Jarrett', 'Peter Overby']
sources = ['_'.join(s.split()) for s in sources]
dic = {}

for source in sources:
    obj_avg = get_avg(source)
    dic[source] = obj_avg

f = open(AVG_FILE, 'w')
f.write('source,average\n')
for key in dic:
    f.write(key + ',' + str(dic[key]) + '\n')

f.close()
print('written to %s' % AVG_FILE)
