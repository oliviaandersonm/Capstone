#apply.py
from sklearn.externals import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
import sys, os, getopt

<<<<<<< HEAD

# def format(input):
#     line = re.sub('[^A-Za-z]', ' ', input)
#     cleaned_line = input.replace(".", "").replace(",", " ").replace("\n", "")
#     return cleaned_line


# article_sentences = []
#
# with open(sys.argv[1]) as article:
#     contents = article.read()
#     contents = re.sub('(Mr.|Dr.|Mrs.|Jr.|Ms.|[A-Z].)', ' ', contents)
#     contents = re.sub('[0-9]\.[0-9]', '', contents)
#     contents = contents.split('.')
#
# line_count = 0
# for line in contents:
#     line_count = line_count + 1
#     line_formatted = format(line)
#     line_inquotes = "\"\"" + line_formatted + "\"\""
#     article_sentences.append(line_inquotes)

#article_sentences = ['']

# download_dir = "input.csv"
#
# csv = open(download_dir, "w")
#
# columnTitleRow = "sentence\n"
# csv.write(columnTitleRow)
#
# for s in article_sentences:
#     row = s + "\n"
#     csv.write(row)
#
# for i in range(1500):
#     csv.write("\"\"placeholder" + str(i) + "\"\"\n")
# csv.close()
=======
#to format lines for tokenizing
def format(input):
    line = re.sub('[^A-Za-z]', ' ', input)
    cleaned_line = input.replace(".", "").replace(",", " ").replace("\n", "")
    return cleaned_line


article_sentences = []

#sys.argv[1] should be a readable (text) file
with open(sys.argv[1]) as article:
    contents = article.read()
    contents = re.sub('(Mr.|Dr.|Mrs.|Jr.|Ms.|[A-Z].)', ' ', contents)
    contents = re.sub('[0-9]\.[0-9]', '', contents)
    contents = contents.split('.')

line_count = 0
for line in contents:
    line_count = line_count + 1
    line_formatted = format(line)
    line_inquotes = "\"\"" + line_formatted + "\"\""
    article_sentences.append(line_inquotes)

article_sentences = ['']

#write split sentences to a temporary file
download_dir = "input.csv"

csv = open(download_dir, "w")

columnTitleRow = "sentence\n"
csv.write(columnTitleRow)

for s in article_sentences:
    row = s + "\n"
    csv.write(row)

#used to scale documents
for i in range(1500):
    csv.write("\"\"placeholder" + str(i) + "\"\"\n")
csv.close()
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18


#load model, shape data, predict
model = joblib.load('test.joblib')
<<<<<<< HEAD
dataset = pd.read_csv(sys.argv[1])
=======
dataset = pd.read_csv("input.csv")
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18

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

to_print = ""
num_subj = 0
num_obj = 0
for s in range(len(sentences)):
    if outputs[s] == 1:
        num_subj += 1
        to_print += (sentences[s] + ". Subjective(1)\n")
    else:
        num_obj += 1
        to_print += (sentences[s]+ ". Objective(0)\n")

print(to_print)
