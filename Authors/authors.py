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
    model = joblib.load('test.joblib')
    dataset = pd.read_csv(csv)
    sentences = []
    for i in range(dataset.shape[0]):
        sentences.append(dataset.iloc[i][0])

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(1,3), max_features=1500)
    X = vectorizer.fit_transform(sentences).toarray()

    from sklearn.feature_extraction.text import TfidfTransformer
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()

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


def get_avg(author):
    print(author)
    sentences = []
    ratios = []
    articles = []
    dataset = pd.read_csv(os.path.expanduser('~/pythonml/CapData/all-the-news/articles1.csv'))
    nyt = dataset.loc[dataset.publication=='New York Times']
    author_data = nyt.loc[nyt.author==author]
    articles = author_data['content'].tolist()

    for i in range(len(articles)):
        articles[i] = re.sub('([0-9]\.[0-9]|Rep\.|Mr\.|Dr\.|Mrs\.|Jr\.|Ms\.|[A-Z]\.)', ' ', articles[i])
        articles[i] = articles[i].replace(',', ' ')
        sentences = articles[i].split('.')
        to_csv(sentences)
        ratio = apply_model('x.csv', len(sentences))
        ratios.append(ratio)


    avg = 0
    for ratio in ratios:
        avg += ratio

    print(avg)
    avg = avg / len(articles)
    print(avg)

    return avg
##############
dataset = pd.read_csv(os.path.expanduser('~/pythonml/CapData/all-the-news/articles1.csv'))
nyt = dataset.loc[dataset.publication=='New York Times']
authors = nyt['author'].unique().tolist()

author_dic = {}
for author in authors:
    temp = nyt.loc[nyt.author==author]
    author_dic[author] = temp.shape[0]

#get rid of authors with < 10 articles
to_delete = []
for key in author_dic:
    if author_dic[key] < 10:
        to_delete.append(key)

for key in to_delete:
    del author_dic[key]

#run
avg_dic = {}
for key in author_dic:
    avg_dic[key] = get_avg(key)

with open('results.csv', 'w') as file:
    file.write('author,avg ratio\n')
    for key in avg_dic:
        file.write('%s,%f' % (key, avg_dic[key]))
