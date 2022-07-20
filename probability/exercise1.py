from collections import defaultdict

from utils import *
from sklearn.metrics import confusion_matrix
import math
import random


def naive_bayes_log():
    percentage_positive_instances_train = 0.2
    percentage_negative_instances_train = 0.2

    percentage_positive_instances_test = 0.2
    percentage_negative_instances_test = 0.2

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train,
                                                      percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    map_positive = {}
    map_negative = {}
    total_positive = 0
    total_negative = 0
    y_true = []
    y_pred = []

    mp_positive = defaultdict(int)
    mp_negative = defaultdict(int)

    for i in range(0, len(pos_train), 1):
        for j in range(0, len(pos_train[i]), 1):
            total_positive += 1

    for i in range(0, len(neg_train), 1):
        for j in range(0, len(neg_train[i]), 1):
            total_negative += 1

    for doc in pos_train:
        for word in doc:
            mp_positive[word] += 1

    for doc in neg_train:
        for word in doc:
            mp_negative[word] += 1

    for e in vocab:
        map_positive[e] = mp_positive[e] / total_positive
        map_negative[e] = mp_negative[e] / total_negative


    true_pos, true_neg = 0, 0

    for i in range(0, len(pos_test), 1):
        y_true.append(1)

        res_pos = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        for j in range(0, len(pos_test[i]), 1):
            if pos_test[i][j] in map_positive.keys():
                if map_positive[pos_test[i][j]] > 0:
                    res_pos = res_pos + math.log10(map_positive[pos_test[i][j]])
            if pos_test[i][j] in map_negative.keys():
                if map_negative[pos_test[i][j]] > 0:
                    res_neg = res_neg + math.log10(map_negative[pos_test[i][j]])

        if res_pos > res_neg:
            y_pred.append(1)
            true_pos += 1
        elif res_pos < res_neg:
            y_pred.append(0)
        else:
            x = random.randint(0, 1)
            if x == 1:
                y_pred.append(1)
                true_pos += 1
            else:
                y_pred.append(0)

    for i in range(0, len(neg_test), 1):
        y_true.append(0)

        res_pos = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        for j in range(0, len(neg_test[i]), 1):
            if neg_test[i][j] in map_positive.keys():
                if map_positive[neg_test[i][j]] > 0:
                    res_pos = res_pos + math.log10(map_positive[neg_test[i][j]])
            if neg_test[i][j] in map_negative.keys():
                if map_negative[neg_test[i][j]] > 0:
                    res_neg = res_neg + math.log10(map_negative[neg_test[i][j]])

        if res_neg > res_pos:
            true_neg += 1
            y_pred.append(0)
        elif res_neg < res_pos:
            y_pred.append(1)
        else:
            x = random.randint(0, 1)
            if x == 0:
                y_pred.append(0)
                true_neg += 1
            else:
                y_pred.append(1)

    print('accuracy with log ' + str((true_pos + true_neg) / (len(pos_test) + len(neg_test))))
    print('precision with log ' + str(true_pos / len(pos_test)))
    print('recall with log ' + str(true_pos / (true_pos + len(neg_test) - true_neg)))
    print(confusion_matrix(y_true, y_pred))


def naive_bayes():
    percentage_positive_instances_train = 0.2
    percentage_negative_instances_train = 0.2

    percentage_positive_instances_test = 0.2
    percentage_negative_instances_test = 0.2

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train,
                                                      percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    map_positive = {}
    map_negative = {}
    total_positive = 0
    total_negative = 0
    y_true = []
    y_pred = []

    mp_positive = defaultdict(int)
    mp_negative = defaultdict(int)

    for i in range(0, len(pos_train), 1):
        for j in range(0, len(pos_train[i]), 1):
            total_positive += 1

    for i in range(0, len(neg_train), 1):
        for j in range(0, len(neg_train[i]), 1):
            total_negative += 1

    for doc in pos_train:
        for word in doc:
            mp_positive[word] += 1

    for doc in neg_train:
        for word in doc:
            mp_negative[word] += 1

    for e in vocab:
        map_positive[e] = mp_positive[e] / total_positive
        map_negative[e] = mp_negative[e] / total_negative

    true_pos, true_neg = 0, 0

    for i in range(0, len(pos_test), 1):
        y_true.append(1)

        res_pos = len(pos_train) / (len(pos_train) + len(neg_train))
        res_neg = len(neg_train) / (len(pos_train) + len(neg_train))

        for j in range(0, len(pos_test[i]), 1):
            if pos_test[i][j] in map_positive.keys():
                res_pos = res_pos * map_positive[pos_test[i][j]]
            else:
                res_pos = 0

            if pos_test[i][j] in map_negative.keys():
                res_neg = res_neg * map_negative[pos_test[i][j]]
            else:
                res_neg = 0

        if res_pos > res_neg:
            y_pred.append(1)
            true_pos += 1
        elif res_pos < res_neg:
            y_pred.append(0)
        else:
            x = random.randint(0, 1)
            if x == 1:
                y_pred.append(1)
                true_pos += 1
            else:
                y_pred.append(0)

    for i in range(0, len(neg_test), 1):
        y_true.append(0)

        res_pos = len(pos_train) / (len(pos_train) + len(neg_train))
        res_neg = len(neg_train) / (len(pos_train) + len(neg_train))

        for j in range(0, len(neg_test[i]), 1):
            if neg_test[i][j] in map_positive.keys():
                res_pos = res_pos * map_positive[neg_test[i][j]]
            else:
                res_pos = 0
            if neg_test[i][j] in map_negative.keys():
                res_neg = res_neg * map_negative[neg_test[i][j]]
            else:
                res_neg = 0

        if res_neg > res_pos:
            true_neg += 1
            y_pred.append(0)
        elif res_neg < res_pos:
            y_pred.append(1)
        else:
            x = random.randint(0, 1)
            if x == 0:
                y_pred.append(0)
                true_neg += 1
            else:
                y_pred.append(1)

    print('accuracy without log ' + str((true_pos + true_neg) / (len(pos_test) + len(neg_test))))
    print('precision without log ' + str(true_pos / len(pos_test)))
    print('recall without log ' + str(true_pos / (true_pos + len(neg_test) - true_neg)))
    print(confusion_matrix(y_true, y_pred))


naive_bayes()
print("\n")
naive_bayes_log()


