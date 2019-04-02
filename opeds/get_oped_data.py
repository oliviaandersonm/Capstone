#get_oped_data.py
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import re
import sys, os
import csv

def to_csv(s):
    download_dir = "x.csv"

    csv = open(download_dir, "w")

    columnTitleRow = "sentence\n"
    csv.write(columnTitleRow)

    for sentence in s:
        row = sentence + "\n"
        csv.write(row)

    for i in range(1500):
        csv.write("\"\"placeholder" + str(i) + "\"\"\n")
    csv.close()

def apply_model(csv, length):
    print("loading model")
    model = joblib.load('test.joblib')
    dataset = pd.read_csv(csv)
    print("read csv")
    sentences = []
    for i in range(dataset.shape[0]):
        sentences.append(dataset.iloc[i][0])

    print("count vectorizer")
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(1,3), max_features=1500)
    X = vectorizer.fit_transform(sentences).toarray()

    print("TfidfTransformer")
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()

    print("predict")
    outputs = model.predict(X)

    num_subj = 0
    num_obj = 0
    for s in range(length):
        if outputs[s] == 1:
            num_subj += 1
        else:
            num_obj += 1

    ratio_obj = num_obj / (num_obj + num_subj)

    return ratio_obj


def get_ratios():
    sentences = []
    ratios = []
    file = open('cnn_opeds.csv')
    file.readline()
    line = file.readline()
    for n in range(4500):
        line = re.sub('([0-9]\.[0-9]|Rep\.|Mr\.|Dr\.|Mrs\.|Jr\.|Ms\.|[A-Z]\.)', ' ', line)
        line = line.replace(',', ' ')
        sentences = line.split('.')
        print("Making csv")
        to_csv(sentences)
        ratio = apply_model('x.csv', len(sentences))
        ratios.append(ratio)
        print("reading next line")
        line = file.readline()

    file.close()
    print("Closed ")

    return ratios

#########
ratios = get_ratios()
print(ratios)
avg = sum(ratios) / len(ratios)
print('AVG: ' + avg)
