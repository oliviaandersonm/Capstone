# Identifying Bias in News Articles - CS Capstone 2019
Working repository project by Emily Zurek & Olivia Anderson - the goal is to use subjectivity/objectivity sentence classification to examine bias in the news.

### Classifiers
`linear-SVC` and `LSTM` contain files to train/test and apply different classification models, built with [scikit-learn](https://scikit-learn.org/stable/) and [Keras](https://keras.io/). `tests` contains some experimentation files and their results. All models were trained and tested using the same data.

#### Data
Train/test data was obtained from [sentence polarity dataset v1.0](http://www.cs.cornell.edu/people/pabo/movie-review-data/), [Sentiment Labelled Sentences Data Set]https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences), and [WikiSplit Dataset](https://github.com/google-research-datasets/wiki-split).  [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) was used with [Google's pre-trained model](https://code.google.com/archive/p/word2vec/). 

### Application/Analysis
`Authors` and `BiasPlot` contain programs to inspect/informally test the classifications on data from [All the news](https://www.kaggle.com/snapcrack/all-the-news).
Files for generating differences and similarities in bodies of text are in `matching`, using [News API](https://newsapi.org/) and Word Mover's Distance as described by [Kusner, Sun, Kolkin, & Weinberger](http://proceedings.mlr.press/v37/kusnerb15.pdf).

