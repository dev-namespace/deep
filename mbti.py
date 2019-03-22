# def prepareData():
from keras.preprocessing.text import Tokenizer
from keras import regularizers
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import hashing_trick
from utils import plotEvolution, getMostCommon, saveNP, loadNP, hashSequence, enforceWordList, oneHotSequence, nHotSequence
import numpy as np
import csv

type2num = {
    'ISTP': 0,
    'ENFJ': 1,
    'ENFP': 2,
    'ENTJ': 3,
    'ENTP': 4,
    'ESFJ': 5,
    'ESFP': 6,
    'ESTJ': 7,
    'ESTP': 8,
    'INFJ': 9,
    'INFP': 10,
    'INTJ': 11,
    'INTP': 12,
    'ISFJ': 13,
    'ISFP': 14,
    'ISTJ': 15,
}

def predictionToLabel(pred):
    for k,v in type2num.items():
        if v == pred.argmax(): return k

def makeWordList(filename, n = 100000):
    words = [w for text in xData for w in text]
    most = getMostUsed(words[:5000000], n).tolist()
    most2 = getMostUsed(words[5000000:10000000], n).tolist()
    most3 = getMostUsed(words[10000000:], n).tolist()
    most_set = set(most)
    most_set.update(most2)
    most_set.update(most3)
    words = np.array(list(most_set))
    with open('datasets/' + filename, 'wb') as outfile:
        np.save(outfile, words)

def getData(filename):
    posts = []
    labels = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            posts.append(row['posts'])
            labels.append(row['type'])
    return posts, labels


types = []
x = []
y = []
# with open('./datasets/mbti_1.csv') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         text = row['posts']
#         tokenized = text_to_word_sequence(text)
#         x.append(tokenized)
#         y.append(type2num[row['type']])
# makeWordList('wordDict_10000.npy', 10000)
# wordList = loadNP('wordDict_10000.npy')
# hashSize = round(len(wordList) * 1.3)
# x = enforceWordList(x, wordList)
# x = hashSequence(x, hashSize)
# x = nHotSequence(x, hashSize)

# saveNP(x, 'x.npy')
# x = loadNP('x.npy')

# 58% accuracy --> falta dropout
posts, labels = getData('./datasets/mbti_1.csv')
t = Tokenizer()
t.fit_on_texts(posts)
x = t.texts_to_matrix(posts, mode='count')
y = list(map(lambda x: type2num[x], labels))
# saveNP(x, 'xtok.npy')
# x = loadNP('xtok.npy')

# ========== we have data

# def train():

from keras import models
from keras import layers

x_train = x[:7500]
y_train = to_categorical(y[:7500])

x_test = x[7500:]
y_test = to_categorical(y[7500:])

x_val = x_train[6500:]
y_val = y_train[6500:]
x_train_partial = x_train[:6500]
y_train_partial = y_train[:6500]

model = models.Sequential()
model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
                       activation='relu', input_shape=(len(x[0]),)))
model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
                       activation='relu'))
model.add(layers.Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train_partial,
                    y_train_partial,
                    epochs=15,
                    batch_size=256,
                    validation_data=(x_val, y_val))

plotEvolution(history)

# train()

# Counts
# [['ENFJ' '190']
#  ['ENFP' '675']
#  ['ENTJ' '231']
#  ['ENTP' '685']
#  ['ESFJ' '42']
#  ['ESFP' '48']
#  ['ESTJ' '39']
#  ['ESTP' '89']
#  ['INFJ' '1470']
#  ['INFP' '1832']
#  ['INTJ' '1091']
#  ['INTP' '1304']
#  ['ISFJ' '166']
#  ['ISFP' '271']
#  ['ISTJ' '205']
#  ['ISTP' '337']]
