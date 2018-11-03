import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import random
from datetime import datetime
from sklearn.metrics import accuracy_score


def euclidean_distance(a, b):
    # a.shape = N x D -> n prototypes of embedding dimensionality (query)
    # b.shape = M x D -> m queries of embedding dimensionality (prototype)
    # ideas on vectorization: https://medium.com/dataholiks-distillery/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    # ideas on vectorization: https://hackernoon.com/speeding-up-your-code-2-vectorizing-the-loops-with-numpy-e380e939bed3
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    # Repeat vector to matrix form
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    # Repeat vector to matrix form
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)


def cosine_distance(a, b):
    # a.shape = N x D -> n prototypes of embedding dimensionality (query)
    # b.shape = M x D -> m queries of embedding dimensionality (prototype)
    # idea: https://stackoverflow.com/questions/48485373/pairwise-cosine-similarity-using-tensorflow?answertab=votes#tab-top
    # normalize each row
    norm_a = tf.nn.l2_normalize(a, axis = 1)
    norm_b = tf.nn.l2_normalize(b, axis = 1)
    # multiply row i with row j using transpose
    # element wise product
    prod = tf.matmul(norm_a, norm_b, adjoint_b = True)
    return 1 - prod


def get_epoch_time():
    return int((datetime.now() - datetime(1970,1,1)).total_seconds())


def label2onehot_exp(label, experiment_classes):
    onehot = np.zeros(len(experiment_classes))
    position = int(np.squeeze(np.where(label==np.array(experiment_classes))))
    onehot[position] = 1
    return onehot


def label2onehot(label, length):
    # example: utils.label2onehot(2,5) > array([0., 0., 1., 0., 0.])
    onehot = np.zeros(length)
    onehot[label] = 1
    return onehot


def onehot2label(gt):
    label = np.int(np.squeeze(np.where(np.array(gt) == max(gt))))
    return label


def count_params(trainable_variables):
    # to return number of trainable variables, specifically: tf.trainable_variables()
    return np.sum([np.prod(v.get_shape().as_list()) for v in trainable_variables])


def load_id2label(gt_file):
    ids = []
    fgt = open(gt_file)
    id2label = dict()
    for line in fgt.readlines():
        id, gt = line.strip().split("\t")
        id2label[id] = onehot2label(eval(gt))
        ids.append(id)
    return ids, id2label


def load_id2gt(gt_file):
    ids = []
    fgt = open(gt_file)
    id2gt = dict()
    for line in fgt.readlines():
        id, gt = line.strip().split("\t")
        id2gt[id] = eval(gt)
        ids.append(id)
    return ids, id2gt


def load_label2ids(id2label):
    # retrieve the ids per class
    label2ids = {}
    for id, label in id2label.items():
        if label in label2ids:
            label2ids[label].append(id)
        else:
            label2ids[label] = [id]
    return label2ids


def load_id2audiopath(index_file):
    f = open(index_file)
    id2audiopath = dict()
    for line in f.readlines():
        id, path = line.strip().split("\t")
        id2audiopath[id] = path
    return id2audiopath


def load_id2audioReprPath(index_file):
    audioReprPaths = []
    fspec = open(index_file)
    id2audioReprPath = dict()
    for line in fspec.readlines():
        id, path, _ = line.strip().split("\t")
        id2audioReprPath[id] = path
        audioReprPaths.append(path)
    return audioReprPaths, id2audioReprPath


def load_id2length(index_file):
    f = open(index_file)
    id2length = dict()
    for line in f.readlines():
        id, length = line.strip().split("\t")
        id2length[id] = int(length)
    return id2length


def accuracy_with_aggergated_predictions(pred_array, id_array, ids, id2label):

        # averaging probabilities -> one could also do majority voting
        y_pred = []
        y_true = []
        for id in ids:
            try:
                avg = np.mean(pred_array[np.where(id_array==id)], axis=0)
                idx_prediction = int(np.where(avg == max(avg))[0][0])
                y_pred.append(idx_prediction)

                label = id2label[id]
                y_true.append(int(label))
            except:
                print(id)

        return accuracy_score(y_true, y_pred)


def few_shot_data_preparation(all_ids_train, all_ids_test, classes_vector, label2ids_train, label2ids_test, config):

    if config['n_shot'] == np.inf:

        ids_train = all_ids_train
        ids_test = all_ids_test
        
        print('Train IDs: ALL!')

        # list of ids corresponding to a class, to compute a prototype
        label2selectedIDs = {}
        for c in classes_vector:
            label2selectedIDs[c] = label2ids_train[c]

    else:

        # for each class, pick N training examples
        first = True
        label2selectedIDs = {}
        for c in classes_vector:

            # training examples: K-examples per class
            if config['n_shot'] != np.inf:
                ids_class_train = random.sample(label2ids_train[c], config['n_shot'])
            else:
                ids_class_train = label2ids_train[c]

            if first:
                ids_train = ids_class_train
                ids_test = label2ids_test[c]
                first = False
            else:
                ids_train = np.concatenate((ids_train,ids_class_train), axis=0)
                ids_test = np.concatenate((ids_test,label2ids_test[c]), axis=0)

            label2selectedIDs[c] = ids_class_train

        print('\nTrain IDs: ' + str(ids_train) + '\n')

    return [ids_train, ids_test, label2selectedIDs]
