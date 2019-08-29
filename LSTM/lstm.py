import pandas as pd
import numpy as np
import sys, os
import re
import nltk
from nltk.corpus import stopwords
import pickle

#constants for options
STEM = False
STOPWORDS = False
RM_PUNCTUATION = True
#constants for model
EMBEDDING_DIM = 300
TEST_SPLIT = 0.2
np.random.seed(7)

#load data
dataset = pd.read_csv(os.path.expanduser('~/Capstone/data/train_test_data/full_data.csv'))

sentences = []
#clean
for i in range(dataset.shape[0]):
    sentence = (dataset.iloc[i][0])
    sentence = sentence.lower()
    #remove numbers
    sentence = ''.join([w for w in sentence if not w.isdigit()])
    #remove contractions
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    #remove punctuation
    if RM_PUNCTUATION:
        sentence = re.sub("[!@#$%+*:()'-]", ' ', sentence)
    #remove stopwords
    if STOPWORDS:
        sw = [w for w in sentence.split() if w not in stopwords.words('english')]
        sentence = ' '.join(sw)
    #use word stems
    if STEM:
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer('english')
        st = [stemmer.stem(w) for w in sentence.split()]
        sentence = ' '.join(st)

    sentences.append(sentence)

#get length of longest sentence
max_length = max([len(s.split()) for s in sentences])
print('Max length: %d' % max_length)

#tokenize train/test data and pad sequences
print('Tokenizing/Padding')
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

from keras.preprocessing.sequence import pad_sequences
sequences = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index
print('%d unique tokens' % len(word_index))
padded = pad_sequences(sequences, maxlen=max_length)
labels = dataset['label'].values
print('Sentence tensor: (%d, %d)' % padded.shape)
print('Label tensor: (%d)' % labels.shape)

#write embeddings to csv
write_file = open('temp_dict.csv', 'w')
write_file.write('word, cfs\n')
read_file = open(os.path.expanduser('~/Capstone/data/wmd/google_vectors.txt'), encoding="utf-8")
for line in read_file:
    vals = line.split()
    word = re.sub(',','',vals[0]).lower()
    cfs = 'x'.join(vals[1:])
    write_file.write('%s,%s\n' % (word, cfs))
read_file.close()
write_file.close()

#read in embeddings as dataframe
df = pd.read_csv('temp_dict.csv')

#map tokenized dictionary to word2vec embeddings
#create matrix based on embedding dimension and fill in with values
num_words = len(word_index) + 1
w2v_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > num_words:
        continue
    try:
        vec = df.loc[df['word'] == word].iloc[:,1].values[0].split('x')
    except(IndexError):
        print('%s not found: no. %d' % (word, i))
    embedding_vector = np.array(vec, dtype=np.float32)
    if embedding_vector is not None:
        w2v_matrix[i] = embedding_vector
print(num_words)

###LSTM model with pre-trained embeddings
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
model = Sequential()
emb_layer = Embedding(num_words,
                      EMBEDDING_DIM,
                      weights=[w2v_matrix],
                      input_length=max_length,
                      trainable=False)

model.add(emb_layer)
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='sigmoid'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
from keras import optimizers
sgd = optimizers.SGD(lr=0.9)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


#split train/test data
arr = np.arange(padded.shape[0])
np.random.shuffle(arr)
padded = padded[arr]
labels = labels[arr]
num_test = int(TEST_SPLIT * padded.shape[0])

X_train = padded[:-num_test]
y_train = labels[:-num_test]
X_test = padded[-num_test:]
y_test = labels[-num_test:]

#train
model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test), verbose=2)

# save model
#model.save('short_Gw2v_LSTM.h5')
#Save tokenizer
#with open('short_lstm_gw2v_tokenizer.pickle', 'wb') as f:
    #pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
