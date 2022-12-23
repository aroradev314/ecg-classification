import math

import numpy as np
import scipy.signal
from keras import Model, Sequential
from keras.layers import Input, Dense
import tensorflow as tf
import matplotlib.pyplot as plt
import wfdb
import pickle
import pywt
from sklearn import metrics
from scipy.signal import find_peaks


def make_windows(data, labels, length, step):
    new_data = []
    new_labels = []

    for i in range(len(data)):
        for j in range(0, len(data[i]) - length + 1, step):
            new_data.append(data[i][j: j + length])
            new_labels.append(labels[i])

    return np.array(new_data), np.array(new_labels)


def load_raw_data(df, sampling_rate, path):
    # Loading all data with signal and meta information
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]

    # Eliminating meta information. We are selecting only signal value of 12 leads
    data = np.array([signal for signal, meta in data])
    return data


def plot(record, title, size):
    discrete_x = [i + 1 for i in range(len(record))]
    plt.figure(figsize=size)
    plt.title(title)
    plt.plot(discrete_x, record)  # zero-indexed
    plt.show()


def basic_model(activation, shape):
    inputs = Input(shape=shape)
    hidden1 = Dense(round(shape / 2), activation=tf.nn.relu)(inputs)  # amount of neurons in hidden layer is mean between
    # first and last layer
    hidden2 = Dense(round(shape / 2), activation=tf.nn.relu)(hidden1)
    outputs = Dense(1, activation=activation)(hidden2)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def to_numpy(array_like):
    return np.array(list(map(lambda x: np.asarray(x), array_like)))


def pickler(file, filename):
    with open(filename, 'wb') as f:
        pickle.dump(file, f)


def unpickler(filename):
    with open(filename, 'rb') as f:
        a = pickle.load(f)
    return a


def mex_func(coef, start, end, a, x):
    A = 2 / (math.sqrt(3 * a) * (math.pi ** 0.25))
    final = []
    for i in range(start, end):
        final.append(coef * (A * (1 - ((x - i) / a) ** 2) * math.exp(-0.5 * ((x - i) / a) ** 2)))
    return np.array(final)


def easy_cwt(data, scales, wavelet):
    final = []
    for i in data:
        final.append(pywt.cwt(i, scales, wavelet)[0])

    return np.array(final)


def plot_peaks(data, shape, height=None, threshold=None, distance=None, prominence=None):
    peaks = find_peaks(data, height=height, threshold=threshold, distance=distance, prominence=prominence)[0]

    values = []
    for i in peaks:
        values.append(data[i])
    plt.figure(figsize=shape)
    plt.plot(np.arange(len(data)), data)
    plt.plot(peaks, values)


def evaluate_model(model_pred, model_true):
    model_pred = np.round(model_pred)

    print(metrics.classification_report(model_true, model_pred))


def one_hot_encoder(labels, classes):
    converted = [[0 for i in range(classes)] for j in labels]
    for i in range(len(labels)):
        converted[i][labels[i]] = 1

    return np.array(converted)


