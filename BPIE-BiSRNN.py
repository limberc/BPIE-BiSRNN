'''
The application of paper LONG TEXT ANALYSIS USING SLICED RECURRENT NEURAL NETWORK WITH BREAKING POINT INFORMATION ENRICHMENT


A simplified version

By [Limber](https://github.com/limberc) and [DeePBluE](https://github.com/DeePBluE666)
'''


import gc
import os
import random

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.activations import softmax
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
from keras.layers import (GRU, LSTM, Activation, Bidirectional, Conv1D, Conv2D,
                          CuDNNGRU, CuDNNLSTM, Dense, Dropout, Embedding,
                          Flatten, GlobalAveragePooling1D, GlobalMaxPooling1D,
                          Input, Lambda, Layer, MaxPooling1D, MaxPooling2D,
                          Permute, Reshape, SpatialDropout1D, TimeDistributed,
                          constraints, initializers, regularizers)
from keras.layers.merge import (Average, Concatenate, Dot, Maximum, Multiply,
                                Subtract, concatenate)
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import gensim
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#---------------prepare dataset-----------------------#
train = pd.read_csv('/home/data/yelp_review_polarity/train.csv', header=None)
test = pd.read_csv('/home/data/yelp_review_polarity/test.csv', header=None)

df = pd.concat([train, test])
df.index = range(df.shape[0])

df = pd.concat([train, test])
df.index = range(df.shape[0])

df['text'] = df[1]

df.fillna(0, inplace=True)
df['text'] = df['text'].apply(str)

Y = df[0].values-1
Y = to_categorical(Y, num_classes=2)
X = df['text'].values

max_num_words = 30000
embedding_dim = 300
num_filters = 64
max_length = 512
batch_size = 512
epoch = 20


x_train = X[:train.shape[0]]
y_train = Y[:train.shape[0]]
x_val = X[train.shape[0]:]
y_val = Y[train.shape[0]:]

tokenizer1 = Tokenizer(num_words=max_num_words)
tokenizer1.fit_on_texts(df['text'])
vocab = tokenizer1.word_index

x_train_word_ids = tokenizer1.texts_to_sequences(x_train)
x_val_word_ids = tokenizer1.texts_to_sequences(x_val)

x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=max_length)
x_val_padded_seqs = pad_sequences(x_val_word_ids, maxlen=max_length)

#-------------------------BPIE Bidirectional operation-------------------------#
m = 5  # add length
n = 16  # number of slice


# Forward
x_val_Forward = []
for i in range(x_val_padded_seqs.shape[0]):
    slice = np.split(x_val_padded_seqs[i], n)
    slice_new = []
    for k in range(n):
        if k == 0:
            slice_temp = [0]*m + list(slice[k])
        else:
            slice_temp = list(slice[k-1])[-m:] + list(slice[k])
        slice_new.append(np.array(slice_temp))
    x_val_Forward.append(slice_new)

x_train_Forward = []
for i in range(x_train_padded_seqs.shape[0]):
    slice = np.split(x_train_padded_seqs[i], n)
    slice_new = []
    for k in range(n):
        if k == 0:
            slice_temp = [0]*m + list(slice[k])
        else:
            slice_temp = list(slice[k-1])[-m:] + list(slice[k])
        slice_new.append(np.array(slice_temp))
    x_train_Forward.append(slice_new)

# Backward
x_val_Backward = []
for i in range(x_val_padded_seqs.shape[0]):
    slice = np.split(x_val_padded_seqs[i], m)
    slice_new = []
    for k in range(m):
        if k == (m-1):
            slice_temp = list(slice[k]) + [0]*n
        else:
            slice_temp = list(slice[k]) + list(slice[k+1])[:n]
        slice_temp = np.array(slice_temp)[::-1]
        slice_new.append(slice_temp)
    x_val_Backward.append(slice_new)


x_train_Backward = []
for i in range(x_train_padded_seqs.shape[0]):
    slice = np.split(x_train_padded_seqs[i], m)
    slice_new = []
    for k in range(m):
        if k == (m-1):
            slice_temp = list(slice[k]) + [0]*n
        else:
            slice_temp = list(slice[k]) + list(slice[k+1])[:n]
        slice_temp = np.array(slice_temp)[::-1]
        slice_new.append(slice_temp)
    x_train_Backward.append(slice_new)

#---------------------------get word embedding--------------------------#
glove_path = '/home/glove/glove.840B.300d.txt'
embeddings_index = {}
f = open(glove_path)
for line in f:
    values = line.split()
    word = values[0]
    try:
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    except:
        try:
            coefs = np.asarray(values[2:], dtype='float32')
            embeddings_index[word] = coefs
        except:
            print("[!] wrong line `{}`".format(word))
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((max_num_words + 1, EMBEDDING_DIM))

for word, i in vocab.items():
    if i < max_num_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be random initialized.
            embedding_matrix[i] = embedding_vector
        else:
            print('no char   %s' % word)


#------------------Build BPIE-BiSRNN model------------------#
def get_model():

    embedding_layer = Embedding(max_num_words + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=32+m,
                                trainable=True)

    # build model
    # forward DR
    input1 = Input(shape=(n,))
    input2 = Input(shape=(n,))

    embed_f = embedding_layer(input1)
    embed_f = SpatialDropout1D(0.25)(embed_f)

    gru1_f = GRU(num_filters, recurrent_activation='sigmoid',
                 activation='elu', return_sequences=False)(embed_f)

    Encoder1 = Model(input1, gru1_f)

    # backward DR
    input2 = Input(shape=(n,))

    embed_b = embedding_layer(input2)
    embed_b = SpatialDropout1D(0.25)(embed_b)

    gru1_b = GRU(num_filters, recurrent_activation='sigmoid',
                 activation='elu', return_sequences=False)(embed_b)

    Encoder2 = Model(input2, gru1_b)

    input3 = Input(shape=(n, 32+m))
    embed2 = TimeDistributed(Encoder1)(input3)

    input4 = Input(shape=(n, 32+m))
    embed3 = TimeDistributed(Encoder2)(input4)

    gru2 = GRU(num_filters, recurrent_activation='sigmoid',
               activation='elu', return_sequences=False)(embed2)

    gru3 = GRU(num_filters, recurrent_activation='sigmoid',
               activation='elu', return_sequences=False)(embed3)

    merged = concatenate([gru2, gru3])

    preds = Dense(2, activation='softmax')(merged)

    model = Model([input3, input4], preds)

    return model


model = get_model()
model.summary()

#----------------------------------Training------------------------------#
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])

best = 'PBIE-BiSRNN.h5'
checkpoint = ModelCheckpoint(
    best, monitor='val_acc', save_best_only=True, mode='max')
callbacks = [checkpoint]

model.fit([np.array(x_train_Forward), np.array(x_train_Backward)], y_train,
          validation_data=(
              [np.array(x_val_Forward), np.array(x_val_Backward)], y_val),
          epochs=epochs,
          batch_size=batch_size,
          callbacks=callbacks,
          verbose=1)
