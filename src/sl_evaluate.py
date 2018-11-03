import argparse
import json
import pescador
import config_file, shared, sl_train
import numpy as np
import tensorflow as tf

DATASET = 'asc' # 'us8k' or 'asc'
TEST_BATCH_SIZE = 128
FILE_INDEX = config_file.DATA_FOLDER + 'audio_representation/'+DATASET+'__time-freq/index.tsv'
# FILE_GROUND_TRUTH_TEST defined in line 34

if __name__ == '__main__':

    # which experiment we want to evaluate?
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help='Id of the configuration dictionary')
    args = parser.parse_args()
    experiment = args.experiment
    experiment_folder = config_file.DATA_FOLDER + 'experiments/' + str(experiment) + '/'
    config = json.load(open(experiment_folder + 'config.json'))
    print('Experiment: ' + str(experiment))
    print('\n' + str(config))

    # load all audio representation paths
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2audioReprPath(FILE_INDEX)

    # load ground truth
    FILE_GROUND_TRUTH_TEST = config_file.DATA_FOLDER + 'index/'+DATASET+'/gt_'+DATASET+'_fold'+str(config['fold'])+'_test.tsv'
    [ids, id2gt] = shared.load_id2gt(FILE_GROUND_TRUTH_TEST)
    [_, id2label] = shared.load_id2label(FILE_GROUND_TRUTH_TEST)
    print(FILE_GROUND_TRUTH_TEST)

    # pescador: define (finite, batched & parallel) streamer
    pack = [config, 'overlap_sampling', 1]
    streams = [pescador.Streamer(sl_train.data_gen, id, id2audio_repr_path[id], id2gt[id], pack) for id in ids]
    mux_stream = pescador.StochasticMux(streams, n_active=TEST_BATCH_SIZE*2, rate=None, mode='exhaustive')
    batch_streamer = pescador.Streamer(pescador.buffer_stream, mux_stream, buffer_size=TEST_BATCH_SIZE)
    #batch_streamer = pescador.ZMQStreamer(batch_streamer)

    # tensorflow: define model and cost
    [x, y_, is_train, y, normalized_y, cost] = sl_train.tf_define_model_and_cost(config)

    # tensorflow: compute the accuracy of each model
    accuracies = []
    fgt = open(experiment_folder + 'models.list')
    for model_name in fgt.readlines():
        print(model_name)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        results_folder = experiment_folder + model_name.strip() + '/'
        saver.restore(sess, results_folder)
        tf_vars = [sess, normalized_y, cost, x, y_, is_train]
        acc, val_cost = sl_train.evaluation(batch_streamer, id2label, ids, tf_vars)
        tr = open(results_folder + 'test.result', 'w')
        tr.write("Accuracy: " + str(acc))
        tr.close()
        print('Accuracy: ' + str(acc))
        accuracies.append(acc)
        sess.close()

    # store experiment results
    print('\nExperiment: ' + str(experiment))
    print(config)
    print('Accuracies: ' + str(accuracies))
    print('Max: ' + str(np.max(accuracies)))
    print('Mean: ' + str(np.mean(accuracies)) + ' +- ' + str(np.std(accuracies)))
    to = open(experiment_folder + 'experiment.result', 'w')
    to.write('Experiment: ' + str(experiment))
    to.write('\nAccuracies: ' + str(accuracies))
    to.write('\nMax: ' + str(np.max(accuracies)))
    to.write('\nMean: ' + str(np.mean(accuracies)) + ' +- ' + str(np.std(accuracies)))
    to.close()
