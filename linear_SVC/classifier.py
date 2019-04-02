<<<<<<< HEAD
=======
#classifier.py
#train and test a linear SVM
#model is saved to LinearSVC.joblibs
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18
import pandas as pd
import numpy as np
import sys, os
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.tokenize import word_tokenize as wt
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

<<<<<<< HEAD
#load data
dataset = pd.read_csv(os.path.expanduser('~/pythonml/CapData/data/full_data.csv'))
=======
#load data (this is not in the repo)
#dataset should be downloaded, can replace with relevant file path
dataset = pd.read_csv(os.path.expanduser('~/Capstone/data/train_test_data/full_data.csv'))
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18


sentences = []
#clean
for i in range(dataset.shape[0]):
    sentence = (dataset.iloc[i][0])
    sentence = re.sub('[^A-Za-z]', ' ', sentence)
    sentence = sentence.lower()
    split_sentence = wt(sentence)

    processed = []
    for word in split_sentence:
        processed.append(stemmer.stem(word))

    cleaned = " ".join(processed)
    sentences.append(cleaned)

print('sentences processed\n')
<<<<<<< HEAD
#test different classifiers
=======
#test different classifiers --- commented out sections can be ignored and are used for testing only
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18
classifiers = []
names = []
#SGDClassifier
# from sklearn.linear_model import SGDClassifier
# SGD_classifier = SGDClassifier()
# classifiers.append(SGD_classifier)
# names.append('SDGClassifier')
#SVM
# from sklearn import svm
# SVM_classifier = svm.SVC()
# classifiers.append(SVM_classifier)
# names.append('SVM')
#GaussianNB
# from sklearn.naive_bayes import GaussianNB
# GNB_classifier = GaussianNB()
# classifiers.append(GNB_classifier)
# names.append('GaussianNaiveBayes')
#MulitnomialNB
# from sklearn.naive_bayes import MultinomialNB
# MNB_classifier = MultinomialNB()
# classifiers.append(MNB_classifier)
# names.append('MultinomialNaiveBayes')
#KNeighbors
# from sklearn.neighbors import KNeighborsClassifier
# KNN_classifier = KNeighborsClassifier(n_neighbors=3)
# classifiers.append(KNN_classifier)
# names.append('KNeighbors')
<<<<<<< HEAD
=======

>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18
#Linear SVC
from sklearn.svm import LinearSVC
LSVC_classifier = LinearSVC(random_state=0)
classifiers.append(LSVC_classifier)
names.append('LinearSVC')

<<<<<<< HEAD
print('added classifiers and names to lists\n')
=======
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18
###############
print('count vectorizer')
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1,3), max_features=1500)
vectorizer.fit(sentences)
X = vectorizer.fit_transform(sentences).toarray()
y = np.array(dataset["label"])

print('tfidf')
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

<<<<<<< HEAD
=======
#loop is if multiple classsifiers are being trained
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18
for i in range(len(classifiers)):

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    #print results
    print(names[i] + 's\n')
    classifiers[i].fit(X_train, y_train)
    y_pred = classifiers[i].predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

    #k-fold cross validation
    from sklearn.model_selection import cross_val_score
    print("k-fold validation")
    scores = cross_val_score(classifiers[i], X, y, cv=5)
    print(scores)

    # save model
    from sklearn.externals import joblib
    joblib.dump(classifiers[i], '%s.joblib' % names[i])
<<<<<<< HEAD

##visualize
from sklearn.feature_selection import RFE
rfe = RFE(LSVC_classifier, 1100)
rfe = rfe.fit(X, y)
with open('rfe_out.txt', 'w') as f:
    f.write(str(rfe.support_))
    f.write(str(rfe.ranking_))
=======
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18
