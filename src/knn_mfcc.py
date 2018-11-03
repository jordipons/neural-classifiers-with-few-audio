import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import shared

FOLD = 0 # which fold?

DATA_FOLDER = '/home/jpons/Github/telefonica/data/'

## ASC dataset ##
AUDIO_FOLDER = '/home/jpons/audio/ASC-TUT/'
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
    'n_shot': 20,
    'num_experiment_runs': 5,
    'audio_folder': AUDIO_FOLDER,
    'data_folder': DATA_FOLDER,
    'index_file': INDEX_FILE,
    'gt_train': GT_TRAIN,
    'gt_test': GT_TEST,
    'num_classes_dataset': NUM_CLASSES,
    'MFCC_number': 20,
    'metric': 'cosine' # 'cosine' or 'euclidean'
}


def extract_mfcc_features(audio, sampling_rate=12000): 
    src, sr = librosa.load(audio, sr=sampling_rate)
    src_zeros = np.zeros(1024)
    if len(src) < 1024:
        src_zeros[:len(src)] = src
        src = src_zeros
    mfcc = librosa.feature.mfcc(src, sampling_rate, n_mfcc=config['MFCC_number'])
    dmfcc = mfcc[:, 1:] - mfcc[:, :-1]
    ddmfcc = dmfcc[:, 1:] - dmfcc[:, :-1]
    return np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                           np.mean(dmfcc, axis=1), np.std(dmfcc, axis=1),
                           np.mean(ddmfcc, axis=1), np.std(ddmfcc, axis=1)), axis=0)


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
        X = []
        Y = []
        for id in ids_train:
            X.append(extract_mfcc_features(id2audio_path[id]))
            Y.append(id2label_train[id])
        X=np.asarray(X)
        Y=np.asarray(Y)
        print(X.shape)
        print(Y.shape)

        print('Fitting model..')
        neigh = KNeighborsClassifier(n_neighbors=1, metric=config['metric'])
        neigh.fit(X, Y)
 
        print('Evaluate model..')
        errors = []
        pred = []
        gt = []
        for id in ids_test:
            try:
                pred.append(neigh.predict([extract_mfcc_features(id2audio_path[id])]))
                gt.append(id2label_test[id])
            except:
                errors.append(id)
                print('IDs with error: ')
                print(errors)

        accuracies.append(accuracy_score(gt, pred))
        print(accuracies)
        print(config)

    print('Mean accuracies:'+str(np.mean(accuracies)))
    print('STD accuracies:'+str(np.std(accuracies)))
