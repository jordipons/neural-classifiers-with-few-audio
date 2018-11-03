import argparse
import json
import os
import time
import random
import pescador
import config_file, transfer_train, sl_train, shared
import numpy as np
import tensorflow as tf
import models_proto as models


def eval(config, ids, id2audio_repr_path, support_set, id2gt, id2label, tf_vars, vis_vars):

    [id_string, save_latents, track_accuracies, printing, transfer_learning, model_folder] = vis_vars

    if transfer_learning:
        [sess, x, q, log_p_y, emb_q, emb_prototypes] = tf_vars
        pack = [config, 'overlap_sampling', 1]
        eval_streams = [pescador.Streamer(transfer_train.data_gen, id, id2audio_repr_path[id], id2gt[id], pack) for id in ids]
    else:
        [sess, x, q, is_train, log_p_y, emb_q, emb_prototypes] = tf_vars
        pack = [config, 'overlap_sampling', 42] # 42 being a second of audio
        eval_streams = [pescador.Streamer(sl_train.data_gen, id, id2audio_repr_path[id], id2gt[id], pack) for id in ids]

    eval_mux_stream = pescador.ChainMux(eval_streams, mode='exhaustive')
    eval_batch_streamer = pescador.Streamer(pescador.buffer_stream, 
                                                     eval_mux_stream, 
                                                     buffer_size=config['test_batch_size'], 
                                                     partial=True)
    first_eval = True
    count = 0
    for eval_batch in eval_batch_streamer:

        if transfer_learning:
            [probabilities, embeddings, prototypes] = sess.run([log_p_y, emb_q, emb_prototypes], 
                                                      feed_dict={x: support_set, q: np.expand_dims(eval_batch['X'], axis=-1)})
        else:
            [probabilities, embeddings, prototypes] = sess.run([log_p_y, emb_q, emb_prototypes], 
                                                      feed_dict={x: support_set, q: np.expand_dims(eval_batch['X'], axis=-1), is_train: False})
        if first_eval:
            first_eval = False
            pred_array = probabilities
            id_array = eval_batch['ID']
            if save_latents:
                embed_array = embeddings
                gt_array = eval_batch['Y']
        else:
            count = count + 1
            pred_array = np.concatenate((pred_array,probabilities), axis=0)
            id_array = np.append(id_array,eval_batch['ID'])
            if save_latents:
                embed_array = np.concatenate((embed_array,embeddings), axis=0)
                gt_array = np.concatenate((gt_array,eval_batch['Y']), axis=0)

    epoch_acc = shared.accuracy_with_aggergated_predictions(pred_array, id_array, ids, id2label)

    if printing:
        print(id_string+' Number of audios: '+str(len(ids)))
        print(id_string+' Accuracy: '+str(epoch_acc))
        print(id_string+' Prototypes: '+str(prototypes.shape))

    if track_accuracies:
        fac = open(model_folder + 'epoch_accuracies.tsv', 'a')
        fac.write(str(epoch_acc)+'\n')
        fac.close()

    if save_latents:
        print(id_string+' Embed_array: '+str(embed_array.shape))
        print(id_string+' GT: '+str(gt_array.shape))

        np.savez(model_folder + 'embeddings_'+id_string+'.npz', embed_array)
        np.savez(model_folder + 'prototypes.npz', prototypes)
        np.savez(model_folder + 'gt_'+id_string+'.npz', gt_array)
        print('Storing latents for visualization..')

        print('\nPrototypes: ')
        print(prototypes)

    return epoch_acc


def fetch_data(classes_vector, label2selectedIDs, id2audio_repr_path, id2gt, config, transfer_learning=False):

    set_dic = {}
    gt_dic = {}
    id_dic = {}
    minimum_number_of_patches = np.inf
    total_number_of_patches = 0

    for c in classes_vector:

        # pescador: to batch the computations
        preprocess_batch_size = np.min([len(label2selectedIDs[c]), config['preprocess_batch_size']])
        print('Batch size: '+str(preprocess_batch_size))
        print('IDs used for computing the category '+str(c)+' prototype: '+str(label2selectedIDs[c]))
        pack = [config, config['train_sampling'], config['param_train_sampling']]
        if transfer_learning:
            streams = [pescador.Streamer(transfer_train.data_gen, id, id2audio_repr_path[id], id2gt[id], pack) for id in label2selectedIDs[c]]
        else:
            streams = [pescador.Streamer(sl_train.data_gen, id, id2audio_repr_path[id], id2gt[id], pack) for id in label2selectedIDs[c]]
        mux_stream = pescador.ChainMux(streams, mode='exhaustive')
        batch_streamer = pescador.Streamer(pescador.buffer_stream, mux_stream, buffer_size=preprocess_batch_size, partial=True)

        # construct data vectors
        first = True
        gt = []
        for batch in batch_streamer:
            if first:
                class_set = batch['X']
                class_gt = batch['Y']
                class_id = batch['ID']
                first = False
            else:
                class_set = np.concatenate((class_set, batch['X']), axis=0)
                class_gt = np.concatenate((class_gt, batch['Y']), axis=0)
                class_id = np.concatenate((class_id, batch['ID']), axis=0)
        print(class_set.shape)
        print(class_gt.shape)
        print(class_id.shape)
        set_dic[c] = class_set
        gt_dic[c] = class_gt
        id_dic[c] = class_id
        minimum_number_of_patches = min(minimum_number_of_patches, class_set.shape[0])
        total_number_of_patches += class_set.shape[0]

    return [set_dic, gt_dic, id_dic, minimum_number_of_patches, total_number_of_patches]


if __name__ == '__main__':

    start_time = time.time()

    # load config parameters defined in 'config_file.py'
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration',
                        help='ID in the config_file dictionary')
    args = parser.parse_args()
    config = config_file.config_proto[args.configuration]

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

    # load test ground truth
    file_ground_truth_test = config_file.DATA_FOLDER + config['gt_test']
    [all_ids_test, id2gt_test] = shared.load_id2gt(file_ground_truth_test)
    [_, id2label_test] = shared.load_id2label(file_ground_truth_test)
    label2ids_test = shared.load_label2ids(id2label_test)

    # set output according to the experimental setup
    classes_vector = list(range(config['num_classes_dataset']))

    # tensorflow: define the model
    with tf.name_scope('model'):

        # support for training [classes, support, time, freq, channel]
        x = tf.placeholder(tf.float32, [None, None, config['xInput'], config['yInput'], 1])
        # query for training [queries, time, freq, channel]
        q = tf.placeholder(tf.float32, [None, config['xInput'], config['yInput'], 1])
        # output for training
        y_one_hot = tf.placeholder(tf.float32, [None, None])
        # is training? For batchnorm and dropout
        is_train = tf.placeholder(tf.bool)

        # fetching: num_classes, num_support, num_queries
        x_shape = tf.shape(x)
        q_shape = tf.shape(q)
        num_classes, num_support = x_shape[0], x_shape[1]
        num_queries = q_shape[0]

        # embedding the supports
        [emb_x, config] = models.select(tf.reshape(x, [num_classes * num_support, config['xInput'], config['yInput'], 1]), config, is_train)
        emb_dim = tf.shape(emb_x)[-1]
        # computing prototypes
        emb_prototypes = tf.reduce_mean(tf.reshape(emb_x, [num_classes, num_support, emb_dim]), axis=1)

        # embedding queries for training
        [emb_q, config] = models.select(q, config, is_train, reuse=True)
 
        # compute distances between prototypes
        dists_protos = shared.euclidean_distance(emb_prototypes, emb_prototypes)
        dists_protos_cos = shared.cosine_distance(emb_prototypes, emb_prototypes)
        mean_dists_protos = tf.reduce_mean(dists_protos)

        # compute distance wrt. every prototype and convert it to proabilites
        #dists = shared.cosine_distance(emb_q, emb_prototypes)
        dists = shared.euclidean_distance(emb_q, emb_prototypes)
        log_p_y = tf.nn.log_softmax(-dists)

        # compute cost
        ce_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1))

        # compute accuracy
        correct_prediction = tf.equal(tf.argmax(log_p_y, 1), tf.argmax(y_one_hot, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # tensorflow: define optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # needed for batchnorm
    with tf.control_dependencies(update_ops):
        lr = tf.placeholder(tf.float32)
        if config['optimizer'] == 'SGD_clip':
            optimizer = tf.train.GradientDescentOptimizer(lr)
            gradients, variables = zip(*optimizer.compute_gradients(ce_loss))
            gradients_clip, _ = tf.clip_by_global_norm(gradients, 5.0)
            train_step = optimizer.apply_gradients(zip(gradients_clip, variables))

    # save experimental settings
    experiment_id = 'fold_'+str(config_file.FOLD)+'_'+str(shared.get_epoch_time())
    experiment_folder = config_file.DATA_FOLDER + 'experiments/' + str(experiment_id) + '/'
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    config['fold'] = config_file.FOLD
    json.dump(config, open(experiment_folder + 'config.json', 'w'))
    print('\nConfig file saved: ' + str(config))

    print('Number of parameters of the model: ' + str(shared.count_params(tf.trainable_variables()))+'\n')

    sess = tf.InteractiveSession()
    acc_experiments = []
    epochs_experiments = []
    for e in range(config['num_experiment_runs']):

        print('\nEXPERIMENT: '+ str(experiment_id) +' '+ str(e+1)+'/'+str(config['num_experiment_runs']))
        print('-----------------------------------')

        # create training folder
        model_id = 'model_fold' + str(config['fold']) + '_' + str(shared.get_epoch_time()) 
        model_folder = experiment_folder + model_id + '/'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        # writing headers of the train_log.tsv
        fy = open(model_folder + 'train_log.tsv', 'a')
        if config['model_number'] > 10:
            fy.write('Update\tEpoch\tloss\tmean_distance_protos\tmean_gradients\taccuracy\talpha\tbeta\n')
        else:
            fy.write('Update\tEpoch\tloss\tmean_distance_protos\tmean_gradients\taccuracy\n')
        fy.close()

        # saving a list of all the models trained for this experiment
        fall = open(experiment_folder + 'models.list', 'a')
        fall.write(str(model_id) +'\n')
        fall.close()

        # few-shot learning: ids selection
        tmp_ids = shared.few_shot_data_preparation(all_ids_train, all_ids_test, classes_vector, label2ids_train, label2ids_test, config)
        [ids_train, ids_test, label2selectedIDs] = tmp_ids

        # few-shot learning: fetch data
        tmp_data = fetch_data(classes_vector, label2selectedIDs, id2audio_repr_path, id2gt_train,  config)
        [train_set_dic, train_gt_dic, train_id_dic, minimum_number_of_patches, total_number_of_patches] = tmp_data
                 
        # tensorflow: create a session to run the tensorflow graph
        sess.run(tf.global_variables_initializer())

        # training
        mean_distances = []
        epoch = 0
        perfect_epoch = 0
        not_improve = 0
        update = 0
        acc_counter = 0
        define_support = True
        max_acc = -np.inf
        while True:
            update += 1

            # define support and query sets differently for every update
            support_set_ten = []
            first = True
            for c in classes_vector:     
                # randomize data
                idx = np.random.permutation(len(train_set_dic[c]))
                idx_support = [random.choice(np.where(train_id_dic[c]==i)[0]) for i in set(train_id_dic[c])]

                # define a fixed support set
                if define_support:
                    if config['patches_per_prototype'] == np.inf:
                        support_set_idx = idx_support[:minimum_number_of_patches]
                    else:
                        support_set_idx = idx_support[:config['patches_per_prototype']]
                    support_set_ten.append(train_set_dic[c][support_set_idx])
                    print('Defining support set!')

                # define some query set that changes every update
                query_set_idx = idx[:config['max_patches_per_class']]
                if first:
                    first=False
                    query_set_ten = train_set_dic[c][query_set_idx]
                    query_gt_ten = train_gt_dic[c][query_set_idx]
                else:
                    query_set_ten = np.concatenate((query_set_ten, train_set_dic[c][query_set_idx]), axis=0)
                    query_gt_ten = np.concatenate((query_gt_ten, train_gt_dic[c][query_set_idx]), axis=0)

            if define_support:
                support_set = np.expand_dims(support_set_ten,axis=-1)
                define_support = False
                print(support_set.shape)

            # format for tensorflow: as np.array and channels last
            query_set = np.expand_dims(query_set_ten,axis=-1)
            query_gt = np.array(query_gt_ten)
            print(query_set.shape) 
            print(query_gt.shape)

            # training
            [_, loss, protos_t, train_acc, mean_d, d_pr, d_pr_cos] = sess.run(
                                  [train_step, ce_loss, emb_prototypes, acc, mean_dists_protos, dists_protos, dists_protos_cos],
                                  feed_dict={x: support_set, q: query_set, y_one_hot: query_gt, lr: config['learning_rate'], is_train: True})

            print('%d/ epoch: %d | loss: %g | avg(D_protos): %g | # support: %d | # query: %d | acc: %g' 
                   % (update, epoch, loss, mean_d, support_set.shape[0], query_set.shape[0], train_acc))
            fy = open(model_folder + 'train_log.tsv', 'a')
            if config['model_number'] > 10:
                fy.write('%d\t%d\t%g\t%g\t%g\t%g\t%g\n' % (update, epoch, loss, mean_d, train_acc,
                          sess.run('model/log_learn/alpha:0'), sess.run('model/log_learn/beta:0')))
            else:
                fy.write('%d\t%d\t%g\t%g\t%g\n' % (update, epoch, loss, mean_d, train_acc))
            fy.close()

            # stop criteria (for every epoch)
            if update % (total_number_of_patches//query_set.shape[0]) == 0:
                epoch += 1

                # EVALUATING TRAIN SET # [id_string, save_latents, track_accuracies, printing, transfer learning?, model_folder]
                vis_vars = ['[epoch-eval]', False, True, True, False, model_folder]
                tf_vars = [sess, x, q, is_train, log_p_y, emb_q, emb_prototypes]
                epoch_acc = eval(config, ids_train, id2audio_repr_path, support_set, id2gt_train, id2label_train, tf_vars, vis_vars)

                if epoch_acc <= max_acc:
                    not_improve += 1
                    print('Did not improve! Max accuracy: '+str(max_acc))
                    if not_improve > config['max_accuracy']: # the same number of epochs without improving or having 100% accuracy
                        break

                if epoch_acc > max_acc:
                    max_acc = epoch_acc

                if  epoch > config['max_epochs']:
                    break

        # EVALUATING TRAIN SET AND STORE STUFF # [id_string, save_latents, track_accuracies, printing, transfer learning?, model_folder]
        vis_vars = ['train_set', True, False, True, False, model_folder]
        tf_vars = [sess, x, q, is_train, log_p_y, emb_q, emb_prototypes]
        train_acc = eval(config, ids_train, id2audio_repr_path, support_set, id2gt_train, id2label_train, tf_vars, vis_vars)

        # EVALUATING TEST SET # [id_string, save_latents, track_accuracies, printing, transfer learning?, model_folder]
        vis_vars = ['test_set', True, False, True, False, model_folder] 
        tf_vars = [sess, x, q, is_train, log_p_y, emb_q, emb_prototypes]
        accuracy = eval(config, ids_test, id2audio_repr_path, support_set, id2gt_test, id2label_test, tf_vars, vis_vars)

        acc_experiments.append(accuracy)
        epochs_experiments.append(epoch)
        tr = open(model_folder + 'test.result', 'w')
        tr.write('Accuracy: ' + str(accuracy))
        tr.close()

        print('Accuracies: ')
        print(acc_experiments)

    print('\n' + str(config))
    print('Average accuracy: ' + str(np.mean(acc_experiments)))
    print('Standard deviation accuracy: ' + str(np.std(acc_experiments)))
    print('Epochs: ' + str(epochs_experiments))
    print('Experiment: ' + str(experiment_id))
    print((time.time() - start_time)//60)
    tr = open(experiment_folder + 'test.result', 'w')
    tr.write('Accuracies: ' + str(acc_experiments))
    tr.write('\nMean accuracy: ' + str(np.mean(acc_experiments)))
    tr.write('Standard deviation accuracy: ' + str(np.std(acc_experiments)))
    tr.write('Epochs: '+str(epochs_experiments))
    tr.close()
