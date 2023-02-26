# Vadim Litvinov
import numpy as np
import sys
import math

K = 7
EPOCHS = 20
ETA = 0.1


# scaling from x_min to x_max
def scaling(data, x_min, x_max):
    denom = x_max - x_min
    # case of same min and max, stay the same
    denom[denom == 0] = 1
    return ((data - x_min) / denom)


def numerize(data):
    data = np.char.replace(data, 'W', '0')
    data = np.char.replace(data, 'R', '1')
    data_nor = []
    # to float and normalize in range from 0 to 1
    for sample in data:
        sample = np.fromstring(sample, dtype=float, sep=',')
        data_nor.append(sample)
    data_nor = np.array(data_nor)
    return data_nor


def distance(sample1, sample2):
    distance = 0.0
    for i in range(len(sample1)):
        distance += (sample1[i] - sample2[i])**2
    return math.sqrt(distance)


def calcaluateWinner(d):
    winners = set()
    max_value = max(set(d), key=d.count)
    for candidate in d:
        if d.count(candidate) == d.count(max_value):
            winners.add(candidate)
    if len(winners) == 1:
        return winners.pop()
    else:
        return min(winners)


def knn(train_x, train_y, test_x):
    neighbors_all = []
    for test_sample in test_x:
        dist = []
        for i in range(len(train_x)):
            dist.append((i, distance(test_sample, train_x[i])))
        dist.sort(key=lambda tup: tup[1])
        neighbors = dist[0:K]
        # choose the closest K
        # save only the indexes
        neighbors = [i[0] for i in neighbors]
        neighbors_all.append(neighbors)
    predictions = []
    for neighbors in neighbors_all:
        neighbors_y = [int(train_y[neighbor]) for neighbor in neighbors]
        predictions.append(calcaluateWinner(neighbors_y))
    return predictions


def shuff(train_x, train_y):
    assert len(train_x) == len(train_y)
    permut = np.random.permutation(len(train_x))
    return train_x[permut], train_y[permut]


def perceptron(train_x, train_y, test_x):
    w = []
    # create random weights for every class
    for i in range(len(np.unique(train_y))):
        w.append([np.random.rand() for i in range(len(train_x[0])-1)])
        # add bias to wights
        w[i].append(1)
    w = np.array(w)
    # train
    for e in range(EPOCHS):
        train_x, train_y = shuff(train_x, train_y)
        for x, y in zip(train_x, train_y):
            y = int(y)
            y_hat = int(np.argmax(np.dot(w, x)))
            if y != y_hat:
                w[y, :] = w[y, :] + ETA * x
                w[y_hat, :] = w[y_hat, :] - ETA * x
    # predict
    predictions = []
    for sample in test_x:
        predictions.append(np.argmax(np.dot(w, sample)))
    return predictions


def predPA(x, w, y):
    w_tmp = np.delete(w, y, 0)
    y_hat = int(np.argmax(np.dot(w_tmp, x)))
    if y == 0:
        return y_hat + 1
    elif y == 1 and y_hat == 1:
        return y_hat + 1
    return y_hat


def passiveAggressive(train_x, train_y, test_x):
    w = []
    # create random weights for every class
    for i in range(len(np.unique(train_y))):
        w.append([np.random.rand() for i in range(len(train_x[0]) - 1)])
        # add bias to wights
        w[i].append(1)
    w = np.array(w)
    train_x, train_y = shuff(train_x, train_y)
    for x, y in zip(train_x, train_y):
        y = int(y)
        # need to take out the w of true y from W
        y_hat = predPA(x, w, y)
        tau = max(0,(1.0-np.dot(w[y, :],x)+np.dot(w[y_hat, :],x))) / (2*((np.linalg.norm(x))**2))
        if y != y_hat:
            # update the right class weights
            w[y, :] = w[y, :] + tau * x
            # update the wrong class weights
            w[y_hat, :] = w[y_hat, :] - tau * x
    predictions = []
    for sample in test_x:
        predictions.append(np.argmax(np.dot(w, sample)))
    return predictions


def main():
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    train_x = np.loadtxt(train_x, dtype=str)
    test_x = np.loadtxt(test_x, dtype=str)
    train_y = np.loadtxt(train_y)
    train_x = numerize(train_x)
    test_x = numerize(test_x)
    min_train = train_x.min(axis=0)
    max_train = train_x.max(axis=0)
    train_x = scaling(train_x, min_train, max_train)
    test_x = scaling(test_x, min_train, max_train)
    knn_pred = knn(train_x, train_y, test_x)
    # add bias column
    train_x = np.c_[train_x, np.ones(len(train_x))]
    test_x = np.c_[test_x, np.ones(len(test_x))]
    perceptron_pred = perceptron(train_x, train_y, test_x)
    pa_pred = passiveAggressive(train_x, train_y, test_x)
    # print predictions
    for i in range(len(knn_pred)):
        print(f"knn: {knn_pred[i]}, perceptron: {perceptron_pred[i]}, pa: {pa_pred[i]}")




main()
