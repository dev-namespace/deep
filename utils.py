import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import hashing_trick

datasetPath = 'datasets/'

# Creates a dictionary that indexes every word with a number
def getIndexDict(values):
    value2index = {}
    index2value = {}
    index = 0
    for value in values:
        value2index[value] = index
        index2value[index] = value
        index += 1
    return value2index, index2value

# Gets the n most repeated words in a list
def getMostCommon(words, n = 100000):
    np.array(words)
    unique, counts = np.unique(words, return_counts=True)
    rang = np.argsort(-counts)
    return unique[rang][:n]

# One hot vectorization of integer sequence
def oneHotSequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# @TODO: make it well
def nHotSequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # results[i, sequence] += 1.  # set specific indices of results[i] to 1s
        for j, index in enumerate(sequence):
            results[i, j][index] += 1
    return results

# Plots epochs over validation data
def plotHistory(history, smooth=False):
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    if smooth:
        acc = smooth_curve(acc)
        val_acc = smooth_curve(val_acc)
        loss = smooth_curve(loss)
        val_loss = smooth_curve(val_loss)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# Enforce word list
def enforceWordList(x, wordList):
    new_x = []
    for i, entry in enumerate(x):
        print("[*] Enforcing position: " + str(i))
        new_entry = np.intersect1d(entry, wordList)
        new_x.append(new_entry)
    return new_x

# Hash sequence, remember 1.3 * vocab_size
def hashSequence(x, size):
    new_x = []
    for i, entry in enumerate(x):
        print("[*] Encoding position: " + str(i))
        result = hashing_trick(' '.join(entry), size, hash_function='md5')
        new_x.append(result)
    return new_x

# Normalizes data
def normalizeData(data):
    mean = data.mean(axis=0)
    data -= mean
    std = data.std(axis=0)
    data/= std
    return data

def saveNP(data, filename):
    with open(datasetPath + filename, 'wb') as outfile:
        np.save(outfile, data)

def loadNP(filename):
    with open(datasetPath + filename, 'rb') as infile:
        return np.load(infile)
