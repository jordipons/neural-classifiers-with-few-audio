import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf

import shared
import vggish_input, vggish_slim, vggish_params

FOLD = 0 # which fold?

DATA_FOLDER = '/data/workspace_pons/'

## ASC dataset ##
AUDIO_FOLDER = '/data/ASC/'
INDEX_FILE = 'index/asc/index_asc_folds_all.tsv'
GT_TRAIN = 'index/asc/gt_asc_fold'+str(FOLD)+'_train.tsv'
GT_TEST = 'index/asc/gt_asc_fold'+str(FOLD)+'_test.tsv'
NUM_CLASSES = 15


## US8k dataset ##
# AUDIO_FOLDER = '/data/UrbanSound8K/audio/'
# INDEX_FILE = 'index/us8k/index_us8k_folds_all.tsv'
# GT_TRAIN = 'index/us8k/gt_us8k_fold'+str(FOLD)+'_train.tsv'
# GT_TEST = 'index/us8k/gt_us8k_fold'+str(FOLD)+'_test.tsv'
# NUM_CLASSES = 10


config = {
    'audio_folder': AUDIO_FOLDER,
    'data_folder': DATA_FOLDER,
    'index_file': INDEX_FILE,
    'gt_train': GT_TRAIN,
    'gt_test': GT_TEST,
    'n_shot': 2, # 1, 5, 10, or np.inf
    'num_experiment_runs': 20,
    'num_classes_dataset': NUM_CLASSES,
    'train_batch': 32,
    'test_batch': 64,
    'metric': 'cosine' # 'cosine' or 'euclidean'
}


def extract_audioset_features(ids, id2audio_path, id2label): 
    first_audio = True
    for i in ids:
        if first_audio:
            input_data = vggish_input.wavfile_to_examples(id2audio_path[i])
            ground_truth = np.repeat(id2label[i], input_data.shape[0], axis=0)
            identifiers = np.repeat(i, input_data.shape[0], axis=0)
            first_audio = False
        else:
            tmp_in = vggish_input.wavfile_to_examples(id2audio_path[i])
            input_data = np.concatenate((input_data, tmp_in), axis=0)
            tmp_gt = np.repeat(id2label[i], tmp_in.shape[0], axis=0)
            ground_truth = np.concatenate((ground_truth, tmp_gt), axis=0)
            tmp_id = np.repeat(i, tmp_in.shape[0], axis=0)
            identifiers = np.concatenate((identifiers, tmp_id), axis=0)

    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish_model.ckpt')
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        extracted_feat = sess.run([embedding_tensor], feed_dict={features_tensor: input_data})
        feature = np.squeeze(np.asarray(extracted_feat))

    return [feature, ground_truth, identifiers]


if __name__ == '__main__':

    # list audios to process: according to 'index_file'
    id2audio_path = dict()
    f = open(config['data_folder'] + config['index_file'])
    for line in f.readlines():
        id, audio_path = line.strip().split("\t")
        id2audio_path[id] = config['audio_folder'] + audio_path

    # load training ground truth
    file_ground_truth_train = config['data_folder'] + config['gt_train']
    [all_ids_train, id2gt_train] = shared.load_id2gt(file_ground_truth_train)
    [_, id2label_train] = shared.load_id2label(file_ground_truth_train)
    label2ids_train = shared.load_label2ids(id2label_train)

    # load test ground truth
    file_ground_truth_test = config['data_folder'] + config['gt_test']
    [all_ids_test, id2gt_test] = shared.load_id2gt(file_ground_truth_test)
    [_, id2label_test] = shared.load_id2label(file_ground_truth_test)
    label2ids_test = shared.load_label2ids(id2label_test)

    classes_vector = list(range(config['num_classes_dataset']))

    accuracies = []

    print(config)
    for e in range(config['num_experiment_runs']):
       
        print('\nExperiment ' + str(e) + '\n------------')

        # few-shot learning: data selection/preparation
        tmp_data = shared.few_shot_data_preparation(all_ids_train, all_ids_test, classes_vector, label2ids_train, label2ids_test, config)
        [ids_train, ids_test, label2selectedIDs] = tmp_data

        print('Extracting training features..')
        first_batch = True
        pointer=-1
        for pointer in range(len(ids_train)//config['train_batch']):
            ids = ids_train[(pointer)*config['train_batch']:(pointer+1)*config['train_batch']]
            [x, y, refs] = extract_audioset_features(ids, id2audio_path, id2label_train)
            if first_batch:
                [X, Y, IDS] = [x, y, refs]
                first_batch = False
            else:
                 X = np.concatenate((X, x), axis=0)
                 Y = np.concatenate((Y, y), axis=0)
                 IDS = np.concatenate((IDS, refs), axis=0)

        # remaining data
        ids = ids_train[(pointer+1)*config['train_batch']:]
        if not len(ids) == 0:
            [x, y, refs] = extract_audioset_features(ids, id2audio_path, id2label_train)
            if first_batch:
                [X, Y, IDS] = [x, y, refs]
                first_batch = False
            else:
                X = np.concatenate((X, x), axis=0)
                Y = np.concatenate((Y, y), axis=0)
                IDS = np.concatenate((IDS, refs), axis=0)
        print(X.shape)
        print(Y.shape)
        print(Y)

        print('Fitting model..')
        neigh = KNeighborsClassifier(n_neighbors=1, metric=config['metric'])
        neigh.fit(X, Y)
 
        print('Evaluate model..')
        print('Test examples: ' + str(len(ids_test)))
        pred = []
        identifiers = []
        first_batch = True
        for pointer in range(len(ids_test)//config['test_batch']): 
            ids = ids_test[(pointer)*config['test_batch']:(pointer+1)*config['test_batch']]
            [x, y, refs] = extract_audioset_features(ids, id2audio_path, id2label_test)
            if first_batch:
                [pred, identifiers] = [neigh.predict(x), refs]
                first_batch = False
            else:
                pred = np.concatenate((pred, neigh.predict(x)), axis=0)
                identifiers = np.concatenate((identifiers, refs), axis=0)
          
        # remaining data
        ids = ids_test[(pointer+1)*config['test_batch']:]
        if not len(ids) == 0:
            [x, y, refs] = extract_audioset_features(ids, id2audio_path, id2label_test)
            pred = np.concatenate((pred, neigh.predict(x)), axis=0)
            identifiers = np.concatenate((identifiers, refs), axis=0)

        # agreggating same ID: majority voting
        y_pred = []
        y_true = []
        for id in ids_test:
            y_pred.append(np.argmax(np.bincount(pred[np.where(identifiers==id)]))) # majority voting
            y_true.append(int(id2label_test[id]))

        accuracies.append(accuracy_score(y_true, y_pred))
        print(accuracies)
        print(config)

    print('Mean accuracies:'+str(np.mean(accuracies)))
    print('STD accuracies:'+str(np.std(accuracies)))
