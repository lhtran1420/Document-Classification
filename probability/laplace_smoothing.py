import math
from collections import defaultdict

from utils import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def laplace():
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

    map_positive0 = {}
    map_negative0 = {}

    map_positive1 = {}
    map_negative1 = {}

    map_positive2 = {}
    map_negative2 = {}

    map_positive3 = {}
    map_negative3 = {}

    map_positive4 = {}
    map_negative4 = {}

    map_positive5 = {}
    map_negative5 = {}

    map_positive6 = {}
    map_negative6 = {}

    map_positive7 = {}
    map_negative7 = {}

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

        map_positive0[e] = (mp_positive[e] + 0.0001) / (total_positive + 0.0001*len(vocab))
        map_negative0[e] = (mp_negative[e] + 0.0001) / (total_negative + 0.0001*len(vocab))


        map_positive1[e] = (mp_positive[e] + 0.001) / (total_positive + 0.001 * len(vocab))
        map_negative1[e] = (mp_negative[e] + 0.001) / (total_negative + 0.001 * len(vocab))

        map_positive2[e] = (mp_positive[e] + 0.01) / (total_positive + 0.01 * len(vocab))
        map_negative2[e] = (mp_negative[e] + 0.01) / (total_negative + 0.01 * len(vocab))

        map_positive3[e] = (mp_positive[e] + 0.1) / (total_positive + 0.1 * len(vocab))
        map_negative3[e] = (mp_negative[e] + 0.1) / (total_negative + 0.1 * len(vocab))

        map_positive4[e] = (mp_positive[e] + 1.0) / (total_positive + 1.0 * len(vocab))
        map_negative4[e] = (mp_negative[e] + 1.0) / (total_negative + 1.0 * len(vocab))

        map_positive5[e] = (mp_positive[e] + 10.0) / (total_positive + 10.0 * len(vocab))
        map_negative5[e] = (mp_negative[e] + 10.0) / (total_negative + 10.0 * len(vocab))

        map_positive6[e] = (mp_positive[e] + 100.0) / (total_positive + 100.0 * len(vocab))
        map_negative6[e] = (mp_negative[e] + 100.0) / (total_negative + 100.0 * len(vocab))

        map_positive7[e] = (mp_positive[e] + 1000.0) / (total_positive + 1000.0 * len(vocab))
        map_negative7[e] = (mp_negative[e] + 1000.0) / (total_negative + 1000.0 * len(vocab))

    true_pos = {}
    true_neg = {}
    true_pos[0] = 0
    true_pos[1] = 0
    true_pos[2] = 0
    true_pos[3] = 0
    true_pos[4] = 0
    true_pos[5] = 0
    true_pos[6] = 0
    true_pos[7] = 0

    true_neg[0] = 0
    true_neg[1] = 0
    true_neg[2] = 0
    true_neg[3] = 0
    true_neg[4] = 0
    true_neg[5] = 0
    true_neg[6] = 0
    true_neg[7] = 0

    for i in range(0, len(pos_test), 1):
        y_true.append(1)
        res_pos = {}
        res_neg = {}

        res_pos[0] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[0] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        res_pos[1] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[1] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        res_pos[2] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[2] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        res_pos[3] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[3] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        res_pos[4] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[4] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        res_pos[5] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[5] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        res_pos[6] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[6] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        res_pos[7] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[7] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        for j in range(0, len(pos_test[i]), 1):
            if pos_test[i][j] in map_positive0.keys():
                res_pos[0] = res_pos[0] + math.log10( map_positive0[pos_test[i][j]] )
                res_pos[1] = res_pos[1] + math.log10(map_positive1[pos_test[i][j]])
                res_pos[2] = res_pos[2] + math.log10(map_positive2[pos_test[i][j]])
                res_pos[3] = res_pos[3] + math.log10(map_positive3[pos_test[i][j]])
                res_pos[4] = res_pos[4] + math.log10(map_positive4[pos_test[i][j]])
                res_pos[5] = res_pos[5] + math.log10(map_positive5[pos_test[i][j]])
                res_pos[6] = res_pos[6] + math.log10(map_positive6[pos_test[i][j]])
                res_pos[7] = res_pos[7] + math.log10(map_positive7[pos_test[i][j]])

            else:
                res_pos[0] = res_pos[0] + math.log10( 0.0001 / (total_positive + 0.0001*len(vocab)) )
                res_pos[1] = res_pos[1] + math.log10( 0.001 / (total_positive + 0.001 * len(vocab)))
                res_pos[2] = res_pos[2] + math.log10(0.01 / (total_positive + 0.01 * len(vocab)))
                res_pos[3] = res_pos[3] + math.log10( 0.1 / (total_positive + 0.1 * len(vocab)))
                res_pos[4] = res_pos[4] + math.log10(1.0 / (total_positive + 1.0 * len(vocab)))
                res_pos[5] = res_pos[5] + math.log10(10.0 / (total_positive + 10.0 * len(vocab)))
                res_pos[6] = res_pos[6] + math.log10(100.0 / (total_positive + 100.0 * len(vocab)))
                res_pos[7] = res_pos[7] + math.log10(1000.0 / (total_positive + 1000.0 * len(vocab)))

            if pos_test[i][j] in map_negative0.keys():
                res_neg[0] = res_neg[0] + math.log10( map_negative0[pos_test[i][j]])
                res_neg[1] = res_neg[1] + math.log10(map_negative1[pos_test[i][j]])
                res_neg[2] = res_neg[2] + math.log10(map_negative2[pos_test[i][j]])
                res_neg[3] = res_neg[3] + math.log10(map_negative3[pos_test[i][j]])
                res_neg[4] = res_neg[4] + math.log10(map_negative4[pos_test[i][j]])
                res_neg[5] = res_neg[5] + math.log10(map_negative5[pos_test[i][j]])
                res_neg[6] = res_neg[6] + math.log10(map_negative6[pos_test[i][j]])
                res_neg[7] = res_neg[7] + math.log10(map_negative7[pos_test[i][j]])

            else:
                res_neg[0] = res_neg[0] + math.log10(0.0001 / (total_negative + 0.0001 * len(vocab)))
                res_neg[1] = res_neg[1] + math.log10(0.001 / (total_negative + 0.001 * len(vocab)))
                res_neg[2] = res_neg[2] + math.log10(0.01 / (total_negative + 0.01 * len(vocab)))
                res_neg[3] = res_neg[3] + math.log10(0.1 / (total_negative + 0.1 * len(vocab)))
                res_neg[4] = res_neg[4] + math.log10(1.0 / (total_negative + 1.0 * len(vocab)))
                res_neg[5] = res_neg[5] + math.log10(10.0 / (total_negative + 10.0 * len(vocab)))
                res_neg[6] = res_neg[6] + math.log10(100.0 / (total_negative + 100.0 * len(vocab)))
                res_neg[7] = res_neg[7] + math.log10(1000.0 / (total_negative + 1000.0 * len(vocab)))

        if res_pos[0] > res_neg[0]:
            true_pos[0] += 1
        elif res_pos[0] == res_neg[0]:
            x = random.randint(0, 1)
            if x == 1:
                true_pos[0] += 1

        if res_pos[1] > res_neg[1]:
            true_pos[1] += 1
        elif res_pos[1] == res_neg[1]:
            x = random.randint(0, 1)
            if x == 1:
                true_pos[1] += 1

        if res_pos[2] > res_neg[2]:
            true_pos[2] += 1
        elif res_pos[2] == res_neg[2]:
            x = random.randint(0, 1)
            if x == 1:
                true_pos[2] += 1

        if res_pos[3] > res_neg[3]:
            true_pos[3] += 1
        elif res_pos[3] == res_neg[3]:
            x = random.randint(0, 1)
            if x == 1:
                true_pos[3] += 1

        if res_pos[4] > res_neg[4]:
            y_pred.append(1)
            true_pos[4] += 1
        elif res_pos[4] < res_neg[4]:
            y_pred.append(0)
        else:
            x = random.randint(0, 1)
            if x == 1:
                y_pred.append(1)
                true_pos[4] += 1
            else:
                y_pred.append(0)


        if res_pos[5] > res_neg[5]:
            true_pos[5] += 1
        elif res_pos[5] == res_neg[5]:
            x = random.randint(0, 1)
            if x == 1:
                true_pos[5] += 1

        if res_pos[6] > res_neg[6]:
            true_pos[6] += 1
        elif res_pos[6] == res_neg[6]:
            x = random.randint(0, 1)
            if x == 1:
                true_pos[6] += 1

        if res_pos[7] > res_neg[7]:
            true_pos[7] += 1
        elif res_pos[7] == res_neg[7]:
            x = random.randint(0, 1)
            if x == 1:
                true_pos[7] += 1

    for i in range(0, len(neg_test), 1):
        y_true.append(0)
        res_pos = {}
        res_neg = {}

        res_pos[0] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[0] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        res_pos[1] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[1] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        res_pos[2] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[2] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        res_pos[3] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[3] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        res_pos[4] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[4] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        res_pos[5] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[5] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        res_pos[6] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[6] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))

        res_pos[7] = math.log10(len(pos_train) / (len(pos_train) + len(neg_train)))
        res_neg[7] = math.log10(len(neg_train) / (len(pos_train) + len(neg_train)))


        for j in range(0, len(neg_test[i]), 1):
            if neg_test[i][j] in map_positive0.keys():
                res_pos[0] = res_pos[0] + math.log10( map_positive0[neg_test[i][j]])
                res_pos[1] = res_pos[1] + math.log10( map_positive1[neg_test[i][j]])
                res_pos[2] = res_pos[2] + math.log10( map_positive2[neg_test[i][j]])
                res_pos[3] = res_pos[3] + math.log10(map_positive3[neg_test[i][j]])
                res_pos[4] = res_pos[4] + math.log10(map_positive4[neg_test[i][j]])
                res_pos[5] = res_pos[5] + math.log10(map_positive5[neg_test[i][j]])
                res_pos[6] = res_pos[6] + math.log10(map_positive6[neg_test[i][j]])
                res_pos[7] = res_pos[7] + math.log10(map_positive7[neg_test[i][j]])

            else:
                res_pos[0] = res_pos[0] + math.log10( 0.0001 / (total_positive + 0.0001 * len(vocab)))
                res_pos[1] = res_pos[1] + math.log10(0.001 / (total_positive + 0.001 * len(vocab)))
                res_pos[2] = res_pos[2] + math.log10(0.01 / (total_positive + 0.01 * len(vocab)))
                res_pos[3] = res_pos[3] + math.log10(0.1 / (total_positive + 0.1 * len(vocab)))
                res_pos[4] = res_pos[4] + math.log10(1.0 / (total_positive + 1.0 * len(vocab)))
                res_pos[5] = res_pos[5] + math.log10(10.0 / (total_positive + 10.0 * len(vocab)))
                res_pos[6] = res_pos[6] + math.log10(100.0 / (total_positive + 100.0 * len(vocab)))
                res_pos[7] = res_pos[7] + math.log10(1000.0 / (total_positive + 1000.0 * len(vocab)))

            if neg_test[i][j] in map_negative0.keys():
                res_neg[0] = res_neg[0] + math.log10( map_negative0[neg_test[i][j]])
                res_neg[1] = res_neg[1] + math.log10( map_negative1[neg_test[i][j]])
                res_neg[2] = res_neg[2] + math.log10( map_negative2[neg_test[i][j]])
                res_neg[3] = res_neg[3] + math.log10( map_negative3[neg_test[i][j]])
                res_neg[4] = res_neg[4] + math.log10( map_negative4[neg_test[i][j]])
                res_neg[5] = res_neg[5] + math.log10( map_negative5[neg_test[i][j]])
                res_neg[6] = res_neg[6] + math.log10( map_negative6[neg_test[i][j]])
                res_neg[7] = res_neg[7] + math.log10( map_negative7[neg_test[i][j]])

            else:
                res_neg[0] = res_neg[0] + math.log10(0.0001 / (total_negative + 0.0001 * len(vocab)))
                res_neg[1] = res_neg[1] + math.log10(0.001 / (total_negative + 0.001 * len(vocab)))
                res_neg[2] = res_neg[2] + math.log10(0.01 / (total_negative + 0.01 * len(vocab)))
                res_neg[3] = res_neg[3] + math.log10(0.1 / (total_negative + 0.1 * len(vocab)))
                res_neg[4] = res_neg[4] + math.log10(1.0 / (total_negative + 1.0 * len(vocab)))
                res_neg[5] = res_neg[5] + math.log10(10.0 / (total_negative + 10.0 * len(vocab)))
                res_neg[6] = res_neg[6] + math.log10(100.0 / (total_negative + 100.0 * len(vocab)))
                res_neg[7] = res_neg[7] + math.log10(1000.0 / (total_negative + 1000.0 * len(vocab)))

        if res_neg[0] > res_pos[0]:
            true_neg[0] += 1
        elif res_pos[0] == res_neg[0]:
            x = random.randint(0, 1)
            if x == 0:
                true_neg[0] += 1

        if res_neg[1] > res_pos[1]:
            true_neg[1] += 1
        elif res_pos[1] == res_neg[1]:
            x = random.randint(0, 1)
            if x == 0:
                true_neg[1] += 1

        if res_neg[2] > res_pos[2]:
            true_neg[2] += 1
        elif res_pos[2] == res_neg[2]:
            x = random.randint(0, 1)
            if x == 0:
                true_neg[2] += 1

        if res_neg[3] > res_pos[3]:
            true_neg[3] += 1
        elif res_pos[3] == res_neg[3]:
            x = random.randint(0, 1)
            if x == 0:
                true_neg[3] += 1

        if res_neg[4] > res_pos[4]:
            y_pred.append(0)
            true_neg[4] += 1
        elif res_pos[4] < res_neg[4]:
            y_pred.append(1)
        else:
            x = random.randint(0, 1)
            if x == 0:
                y_pred.append(0)
                true_neg[4] += 1
            else:
                y_pred.append(1)

        if res_neg[5] > res_pos[5]:
            true_neg[5] += 1
        elif res_pos[5] == res_neg[5]:
            x = random.randint(0, 1)
            if x == 0:
                true_neg[5] += 1

        if res_neg[6] > res_pos[6]:
            true_neg[6] += 1
        elif res_pos[6] == res_neg[6]:
            x = random.randint(0, 1)
            if x == 0:
                true_neg[6] += 1

        if res_neg[7] > res_pos[7]:
            true_neg[7] += 1
        elif res_pos[7] == res_neg[7]:
            x = random.randint(0, 1)
            if x == 0:
                true_neg[7] += 1

    print("1.0")
    print('accuracy ' + str((true_pos[4] + true_neg[4]) / (len(pos_test) + len(neg_test))))
    print('precision ' + str(true_pos[4] / len(pos_test)))
    print('recall ' + str(true_pos[4] / (true_pos[4] + len(neg_test) - true_neg[4])))
    print(confusion_matrix(y_true, y_pred))

    y0 = (true_pos[0] + true_neg[0]) / (len(pos_test) + len(neg_test))
    y1 = (true_pos[1] + true_neg[1]) / (len(pos_test) + len(neg_test))
    y2 = (true_pos[2] + true_neg[2]) / (len(pos_test) + len(neg_test))
    y3 = (true_pos[3] + true_neg[3]) / (len(pos_test) + len(neg_test))
    y4 = (true_pos[4] + true_neg[4]) / (len(pos_test) + len(neg_test))
    y5 = (true_pos[5] + true_neg[5]) / (len(pos_test) + len(neg_test))
    y6 = (true_pos[6] + true_neg[6]) / (len(pos_test) + len(neg_test))
    y7 = (true_pos[7] + true_neg[7]) / (len(pos_test) + len(neg_test))

    return y0, y1, y2, y3, y4, y5, y6, y7


x = ["-4", "-3", "-3", "-1", "0", "1", "2", "3"]
y0, y1, y2, y3, y4, y5, y6, y7 = laplace()
y = [y0, y1, y2, y3, y4, y5, y6, y7]
plt.plot(x, y)
plt.show()

