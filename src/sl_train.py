import argparse
import json
import os
import time
import random
import pescador
import numpy as np
import tensorflow as tf
import models_sl as models
import config_file, shared
import pickle
from tensorflow.python.framework import ops


def tf_define_model_and_cost(config):

    # tensorflow: define the model
    with tf.name_scope('model'):
        x = tf.placeholder(tf.float32, [None, config['xInput'], config['yInput']])
        y_ = tf.placeholder(tf.float32, [None, config['num_classes_dataset']])
        is_train = tf.placeholder(tf.bool)
        y = models.model_number(x, is_train, config)
        normalized_y = tf.nn.softmax(y)
    print('Number of parameters of the model: ' + str(shared.count_params(tf.trainable_variables()))+'\n')

    # tensorflow: define cost function
    with tf.name_scope('metrics'):
        # if you use softmax_cross_entropy be sure that the output of your model has linear units!
        cost = tf.losses.softmax_cross_entropy(onehot_labels=y_,logits=y)
        if config['weight_decay'] != None:
            vars = tf.trainable_variables() 
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'kernel' in v.name ])
            cost = cost + config['weight_decay']*lossL2
            print('L2 norm, weight decay!')
     
    return [x, y_, is_train, y, normalized_y, cost]


def pre_processing(audio_rep, n_frames, pad_short, pre_processing, audio_rep_type, normalize_mean, normalize_std):

    # if it is too short: pad!
    length = audio_rep.shape[0]
    if length < n_frames:
        audio_rep = np.squeeze(audio_rep)
        if pad_short == 'zero-pad':
            pad = np.zeros((n_frames - audio_rep.shape[0], audio_rep.shape[1]))
            audio_rep = np.concatenate((pad[:pad.shape[0] // 2], audio_rep))
            audio_rep = np.concatenate((audio_rep, pad[pad.shape[0] // 2:]))
        elif pad_short == 'repeat-pad':
            src_repeat = audio_rep
            while (src_repeat.shape[0] < n_frames):
                src_repeat = np.concatenate((src_repeat, audio_rep), axis=0)    
                audio_rep = src_repeat
                audio_rep = audio_rep[:n_frames, :]

    # spectrogram compression
    if pre_processing == 'logEPS' and audio_rep_type == 'time-freq':
        audio_rep = np.log10(audio_rep + np.finfo(float).eps)
    elif pre_processing == 'log10000' and audio_rep_type == 'time-freq':
        audio_rep = np.log10(10000 * audio_rep + 1)

    # zero-mean and one-var
    if audio_rep_type == 'time-freq' and normalize_mean is not None:
        # default: None (defined in audio_representation.py)
        # to normalize considering some data: run compute_mean_var.py
        audio_rep = (audio_rep - normalize_mean) / normalize_std

    return audio_rep


def data_gen(id, audio_repr_path, gt, pack):

    [config, sampling, param_sampling] = pack

    # load and pre-process audio representation -> audio_repr shape: NxM
    audio_rep = pickle.load(open(config_file.DATA_FOLDER + audio_repr_path, 'rb'))
    
    audio_rep = pre_processing(audio_rep, config['n_frames'], config['pad_short'], config['pre_processing'],
                               config['audio_rep']['type'], config['audio_rep']['normalize_mean'], config['audio_rep']['normalize_std'])

    # let's deliver some data!
    last_frame = int(audio_rep.shape[0]) - int(config['n_frames']) + 1
    if sampling == 'random':
        for i in range(0, param_sampling):
            time_stamp = random.randint(0,last_frame-1)
            yield dict(X = audio_rep[time_stamp : time_stamp+config['n_frames'], : ], Y = gt, ID = id)

    elif sampling == 'overlap_sampling':
        for time_stamp in range(0, last_frame, param_sampling):
            yield dict(X = audio_rep[time_stamp : time_stamp+config['n_frames'], : ], Y = gt, ID = id)


def evaluation(batch_dispatcher, id2label, ids_test, tf_vars):

    [sess, normalized_y, cost, x, y_, is_train] = tf_vars
    array_cost = []
    first = True
    for batch in batch_dispatcher:
        pred = sess.run([normalized_y], feed_dict={x: batch['X'], is_train: False})
        cost_pred = sess.run([cost], feed_dict={x: batch['X'], y_: batch['Y'], is_train: False})
        array_cost.append(cost_pred)
        if first:
            pred_array = pred[0]
            id_array = batch['ID']
            first = False
        else:
            pred_array = np.concatenate((pred_array,pred[0]), axis=0)
            id_array = np.append(id_array,batch['ID'])

    accuracy = shared.accuracy_with_aggergated_predictions(pred_array, id_array, ids_test, id2label)

    return accuracy, np.mean(array_cost)


if __name__ == '__main__':

    # load config parameters defined in 'config_file.py'
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration',
                        help='ID in the config_file dictionary')
    args = parser.parse_args()
    config = config_file.config_sl[args.configuration]

    # load config parameters used in 'audio_representation.py',
    config_json = config_file.DATA_FOLDER + config['audio_representation_folder'] + 'config.json'
    with open(config_json, "r") as f:
        params = json.load(f)
    config['audio_rep'] = params

    # set patch parameters
    config['xInput'] = config['n_frames']
    config['yInput'] = config['audio_rep']['n_mels']

    # load audio representation paths
    file_index = config_file.DATA_FOLDER + config['audio_representation_folder'] + 'index.tsv'
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2audioReprPath(file_index)

    # load training ground truth
    file_ground_truth_train = config_file.DATA_FOLDER + config['gt_train']
    [all_ids_train, id2gt_train] = shared.load_id2gt(file_ground_truth_train)
    [_, id2label_train] = shared.load_id2label(file_ground_truth_train)
    label2ids_train = shared.load_label2ids(id2label_train)

    # set output according to the experimental setup
    config['classes_vector'] = list(range(config['num_classes_dataset']))

    # save experimental settings
    experiment_id = 'fold_'+str(config_file.FOLD)+'_'+str(shared.get_epoch_time())
    experiment_folder = config_file.DATA_FOLDER + 'experiments/' + str(experiment_id) + '/'
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    config['fold'] = config_file.FOLD
    json.dump(config, open(experiment_folder + 'config.json', 'w'))
    print('\nConfig file saved: ' + str(config))

    # tensorflow: define model and cost
    [x, y_, is_train, y, normalized_y, cost] = tf_define_model_and_cost(config)

    # tensorflow: define optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # needed for batchnorm
    with tf.control_dependencies(update_ops):
        lr = tf.placeholder(tf.float32)
        if config['optimizer'] == 'SGD_clip':
            optimizer = tf.train.GradientDescentOptimizer(lr)
            gradients, variables = zip(*optimizer.compute_gradients(cost))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            train_step = optimizer.apply_gradients(zip(gradients, variables))

    for e in range(config['num_experiment_runs']):

        print('\nEXPERIMENT: '+ str(experiment_id) +' '+ str(e+1)+'/'+str(config['num_experiment_runs']))
        print('-----------------------------------')

        # create training folder
        model_id = 'model_fold' + str(config['fold']) + '_' + str(shared.get_epoch_time()) 
        model_folder = experiment_folder + model_id + '/'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        # listing all models trained for this experiment
        fall = open(experiment_folder + 'models.list', 'a')
        fall.write(str(model_id) +'\n')
        fall.close()

        # few-shot learning: data selection/preparation
        dummy_list = []
        if config['num_classes_dataset'] == 10: # dummy dictionary for US8k dataset (to easily share this function)
             dummy_dic = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        elif config['num_classes_dataset'] == 15: # dummy dictionary for ASC dataset (to easily share this function)
             dummy_dic = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
        tmp_data = shared.few_shot_data_preparation(all_ids_train, dummy_list, config['classes_vector'], label2ids_train, dummy_dic, config)
        [ids_train, _, _] = tmp_data

        # pescador train: define streamer
        train_pack = [config, config['train_sampling'], config['param_train_sampling']]
        train_streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], id2gt_train[id], train_pack) for id in ids_train]
        train_mux_stream = pescador.StochasticMux(train_streams, n_active=config['batch_size']*2, rate=None, mode='exhaustive')
        train_batch_streamer = pescador.Streamer(pescador.buffer_stream, train_mux_stream, buffer_size=config['batch_size'], partial=True)
      
        # tensorflow: create a session to run the tensorflow graph
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if config['load_model'] != None: # restore model weights from previously saved model
            saver.restore(sess, config['load_model']) # end with /!
            print('Pre-trained model loaded!')
 
        # writing headers of the train_log.tsv
        fy = open(model_folder + 'train_log.tsv', 'a')
        if config['model_number'] > 10:
            fy.write('Epoch\ttrain_cost\tepoch_time\tlearing_rate\talpha\tbeta\n')
        else:
            fy.write('Epoch\ttrain_cost\tepoch_time\tlearing_rate\n')
        fy.close()

        # training
        cost_best_model = np.Inf
        acc_best_model = 0
        k_patience = 0
        print('Training started..')
        for i in range(config['epochs']):

            # training: do not train first epoch, to see random weights behaviour
            start_time = time.time()
            array_train_cost = []
            if i != 0:
                for train_batch in train_batch_streamer:
                    _, train_cost = sess.run([train_step, cost], 
                                             feed_dict={x: train_batch['X'], y_: train_batch['Y'], lr: config['learning_rate'], is_train: True})
                    array_train_cost.append(train_cost)

            # Keep track of average loss of the epoch
            train_cost = np.mean(array_train_cost)
            epoch_time = time.time() - start_time
            fy = open(model_folder + 'train_log.tsv', 'a')
            if config['model_number'] > 10:
                fy.write('%d\t%g\t%gs\t%g\t%g\t%g\n' % (i+1, train_cost, epoch_time, config['learning_rate'], 
                          sess.run('model/log_learn/alpha:0'), sess.run('model/log_learn/beta:0')))
            else:
                fy.write('%d\t%g\t%gs\t%g\n' % (i+1, train_cost, epoch_time, config['learning_rate']))
            fy.close()

            if config['model_number'] > 10:
                print('Epoch %d, train cost %g, epoch-time %gs, lr %g, time-stamp %s, alpha %g, beta %g' %
                      (i+1, train_cost, epoch_time, config['learning_rate'], 
                      str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())), 
                      sess.run('model/log_learn/alpha:0'), sess.run('model/log_learn/beta:0')))
            else:
                print('Epoch %d, train cost %g, epoch-time %gs, lr %g, time-stamp %s' %
                      (i+1, train_cost, epoch_time, config['learning_rate'], str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))))

        # save model weights to disk
        print('Saving last model..')
        save_path = saver.save(sess, model_folder)
        sess.close()

    print('\nEVALUATE EXPERIMENT -> '+ str(experiment_id))
