import argparse
import json
import os
import time
import math
import random
import pescador
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import pickle

import config_file, shared
import vggish_input, vggish_slim, vggish_params

def data_gen(id, audio_repr_path, gt, pack):

    [config, sampling, param_sampling] = pack

    # load and pre-process audio representation -> audio_repr shape: NxM
    audio_rep = pickle.load(open(config_file.DATA_FOLDER + audio_repr_path, 'rb'))

    # let's deliver some data!
    patches = int(audio_rep.shape[0])
    if sampling == 'random':
        for i in range(0, param_sampling):
            patch = random.randint(0,patches-1)
            yield dict(X = audio_rep[patch,:,:], Y = gt, ID = id)

    elif sampling == 'overlap_sampling':
        for patch in range(0, patches, param_sampling):
            yield dict(X = audio_rep[patch,:,:], Y = gt, ID = id)


def tf_define_model_and_cost(config):

    slim = tf.contrib.slim
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                         weights_initializer=tf.truncated_normal_initializer(stddev=vggish_params.INIT_STDDEV),
                         biases_initializer=tf.zeros_initializer(),
                         activation_fn=tf.nn.relu,
                         trainable=True), \
         slim.arg_scope([slim.conv2d],
                         kernel_size=[3, 3], stride=1, padding='SAME'), \
         slim.arg_scope([slim.max_pool2d],
                         kernel_size=[2, 2], stride=2, padding='SAME'), \
         tf.variable_scope('vggish'):

        x = tf.placeholder(tf.float32, shape=(None, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS), name='input_features')
        net = tf.reshape(x, [-1, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS, 1])
        net = slim.conv2d(net, 64, scope='conv1')
        net = slim.max_pool2d(net, scope='pool1')
        net = slim.conv2d(net, 128, scope='conv2')
        net = slim.max_pool2d(net, scope='pool2')
        net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3')
        net = slim.max_pool2d(net, scope='pool3')
        net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4')
        net = slim.max_pool2d(net, scope='pool4')
        net = slim.flatten(net)
        net = slim.repeat(net, 2, slim.fully_connected, 4096, scope='fc1')
        net = slim.fully_connected(net, vggish_params.EMBEDDING_SIZE, scope='fc2')
        embeddings = tf.identity(net, name='embedding')

    with tf.variable_scope('my_model'):
        y = slim.fully_connected(embeddings, config['num_classes_dataset'], activation_fn=None, scope='logits')
        normalized_y = tf.nn.softmax(y)
        y_ = tf.placeholder(tf.float32, [None, config['num_classes_dataset']])
        cost = tf.losses.softmax_cross_entropy(onehot_labels=y_,logits=y)

    print('Number parameters of the model: ' + str(shared.count_params(tf.trainable_variables()))+'\n')
     
    return [x, y_, y, normalized_y, cost]

def evaluation_audioset(batch_dispatcher, id2label, ids_test, tf_vars):

    [sess, normalized_y, cost, x, y_] = tf_vars
    array_cost = []
    first = True
    for batch in batch_dispatcher:
        [pred, cost_pred] = sess.run([normalized_y, cost], feed_dict={x: batch['X'], y_: batch['Y']})
        array_cost.append(cost_pred)
        if first:
            pred_array = pred
            id_array = batch['ID']
            first = False
        else:
            pred_array = np.concatenate((pred_array,pred), axis=0)
            id_array = np.append(id_array,batch['ID'])
   
    accuracy = shared.accuracy_with_aggergated_predictions(pred_array, id_array, ids_test, id2label)

    return accuracy, np.mean(array_cost)


if __name__ == '__main__':

    # load config parameters from 'config_file.py'
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration',
                        help='ID in the config_file dictionary')
    args = parser.parse_args()
    config = config_file.config_transfer[args.configuration]

    # load config parameters used in 'audio_representation.py'
    config_json = config_file.DATA_FOLDER + config['audio_representation_folder'] + 'config.json'
    with open(config_json, "r") as f:
        params = json.load(f)
    config['audio_rep'] = params

    # load audio representation paths
    file_index = config_file.DATA_FOLDER + config['audio_representation_folder'] + 'index.tsv'
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2audioReprPath(file_index)

    # load training ground truth
    file_ground_truth_train = config_file.DATA_FOLDER + config['gt_train']
    [all_ids_train, id2gt_train] = shared.load_id2gt(file_ground_truth_train)
    [_, id2label_train] = shared.load_id2label(file_ground_truth_train)
    label2ids_train = shared.load_label2ids(id2label_train)

    # load validation ground truth
    file_ground_truth_val = config_file.DATA_FOLDER + config['gt_val']
    [all_ids_val, id2gt_val] = shared.load_id2gt(file_ground_truth_val)
    [_, id2label_val] = shared.load_id2label(file_ground_truth_val)
    label2ids_val = shared.load_label2ids(id2label_val)

    # set outputs according to experimental setup
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
    [x, y_, y, normalized_y, cost] = tf_define_model_and_cost(config)

    # tensorflow: define optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # needed for batchnorm
    with tf.control_dependencies(update_ops):
        lr = tf.placeholder(tf.float32)
        if config['optimizer'] == 'SGD_clip':
            optimizer = tf.train.GradientDescentOptimizer(lr)
            gradients, variables = zip(*optimizer.compute_gradients(cost))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            train_step = optimizer.apply_gradients(zip(gradients, variables))
        elif config['optimizer'] == 'many_lr_audioset':
            MULTIPLY_LR = 0.0001 # used to have a different learning rate for my_model variables, i.e.: 0.0001
            print('Different learning rates: '+str(MULTIPLY_LR))
            opt = tf.train.GradientDescentOptimizer(lr)
            grads_and_vars = opt.compute_gradients(cost)
            trainable = [gv for gv in grads_and_vars if 'my_model' in str(gv[1])]
            to_modify_lr = [gv for gv in grads_and_vars if 'my_model' not in str(gv[1])]
            for t in to_modify_lr: 
                trainable.append((t[0]*MULTIPLY_LR,t[1]))
            train_step = opt.apply_gradients(trainable)

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
        tmp_data = shared.few_shot_data_preparation(all_ids_train, all_ids_val, config['classes_vector'], label2ids_train, label2ids_val, config)
        [ids_train, ids_val, _] = tmp_data

        # pescador train: define streamer
        train_pack = [config, config['train_sampling'], config['param_train_sampling']]
        train_streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], id2gt_train[id], train_pack) for id in ids_train]
        train_mux_stream = pescador.StochasticMux(train_streams, n_active=config['batch_size']*2, rate=None, mode='exhaustive')
        train_batch_streamer = pescador.Streamer(pescador.buffer_stream, train_mux_stream, buffer_size=config['batch_size'], partial=True)
   
        # pescador val: define streamer
        val_batch_size = np.min([len(ids_val), config['val_batch_size']])
        val_pack = [config, 'overlap_sampling', 1]
        val_streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], id2gt_val[id], val_pack) for id in ids_val]
        val_mux_stream = pescador.ChainMux(val_streams, mode='exhaustive')
        val_batch_streamer = pescador.Streamer(pescador.buffer_stream, val_mux_stream, buffer_size=val_batch_size, partial=True)
    
        # tensorflow: create a session to run the tensorflow graph
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        vggish_vars = [v for v in tf.global_variables() if 'my_model' not in v.name]
        for variables in vggish_vars:
            print(variables)

        loader = tf.train.Saver(vggish_vars, name='vggish_load_pretrained', write_version=1)
        loader.restore(sess, 'vggish_model.ckpt')
        saver = tf.train.Saver()
 
        # writing train_log.tsv headers
        fy = open(model_folder + 'train_log.tsv', 'a')
        fy.write('Epoch\ttrain_cost\tval_cost\taccuracy\tepoch_time\tlearing_rate\n')
        fy.close()

        # training
        cost_best_model = np.Inf
        acc_best_model = 0
        k_patience = 0
        tmp_learning_rate = config['learning_rate']
        print('Training started..')
        for i in range(config['epochs']):

            # training: do not train first epoch, to see random weights behaviour
            start_time = time.time()
            array_train_cost = []
            if i != 0:
                for train_batch in train_batch_streamer:
                    _, train_cost = sess.run([train_step, cost], 
                                             feed_dict={x: train_batch['X'], y_: train_batch['Y'], lr: tmp_learning_rate})
                    array_train_cost.append(train_cost)

            # validation
            tf_vars = [sess, normalized_y, cost, x, y_]
            accuracy, val_cost = evaluation_audioset(val_batch_streamer, id2label_val, ids_val, tf_vars)

            # Keep track of average loss of the epoch
            train_cost = np.mean(array_train_cost)
            epoch_time = time.time() - start_time
            fy = open(model_folder + 'train_log.tsv', 'a')
            fy.write('%d\t%g\t%g\t%g\t%gs\t%g\n' % (i+1, train_cost, val_cost, accuracy, epoch_time, tmp_learning_rate))
            fy.close()
                 
            # Early stopping: keep the best model in validation set
            if acc_best_model > accuracy:
                print('Epoch %d, train cost %g, val cost %g, accuracy %g, '
                      'epoch-time %gs, lr %g, time-stamp %s' %
                      (i+1, train_cost, val_cost, accuracy, epoch_time, tmp_learning_rate,
                       str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))))
                k_patience += 1  
            
            else:
                # save model weights to disk
                save_path = saver.save(sess, model_folder)
                print('Epoch %d, train cost %g, val cost %g, accuracy %g, '
                      'epoch-time %gs, lr %g, time-stamp %s\n - [BEST MODEL]'
                      ' saved in: %s' %
                      (i+1, train_cost, val_cost, accuracy, epoch_time, tmp_learning_rate,
                       str(time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())), save_path))
                cost_best_model = val_cost
                acc_best_model = accuracy
        sess.close()
    print('\nEVALUATE EXPERIMENT -> '+ str(experiment_id))
