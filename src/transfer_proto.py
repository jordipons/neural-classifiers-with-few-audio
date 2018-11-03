import argparse
import json
import os
import time
import random
import config_file, shared, transfer_train, vggish_params, proto
import numpy as np
import tensorflow as tf


def audioset_model(input_signal, reuse=False):

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
         tf.variable_scope('vggish', reuse=reuse):

        net = slim.conv2d(input_signal, 64, scope='conv1')
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

    with tf.variable_scope('my_model', reuse=reuse):
        return slim.fully_connected(embeddings, 10, activation_fn=None, scope='logits')


if __name__ == '__main__':

    start_time = time.time()

    # load config parameters defined in 'config_file.py'
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration',
                        help='ID in the config_file dictionary')
    args = parser.parse_args()
    config = config_file.config_transfer_proto[args.configuration]

    # load config parameters used in 'audio_representation.py',
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
        x = tf.placeholder(tf.float32, [None, None, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS, 1])
 
        # query for training [queries, time, freq, channel]
        q = tf.placeholder(tf.float32, [None, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS, 1])

        # fetching: num_classes, num_support, num_queries
        x_shape = tf.shape(x)
        q_shape = tf.shape(q)
        num_classes, num_support = x_shape[0], x_shape[1]
        num_queries = q_shape[0]

        # embedding the supports
        emb_x = audioset_model(tf.reshape(x, [-1, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS, 1]))
        emb_dim = tf.shape(emb_x)[-1]

        # computing prototypes
        emb_prototypes = tf.reduce_mean(tf.reshape(emb_x, [num_classes, -1, emb_dim]), axis=1)

        # embedding queries for training
        emb_q = audioset_model(q, reuse=True)
 
        # compute distance wrt. every prototype and convert it to proabilites
        #dists = shared.cosine_distance(emb_q, emb_prototypes)
        dists = shared.euclidean_distance(emb_q, emb_prototypes)
        log_p_y = tf.nn.log_softmax(-dists)

        # compute distances between prototypes
        dists_protos = shared.euclidean_distance(emb_prototypes, emb_prototypes)
        dists_protos_cos = shared.cosine_distance(emb_prototypes, emb_prototypes)
        mean_dists_protos = tf.reduce_mean(dists_protos)

        # compute cost
        y_one_hot = tf.placeholder(tf.float32, [None, None])
        ce_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1))

        # compute accuracy
        correct_prediction = tf.equal(tf.argmax(log_p_y, 1), tf.argmax(y_one_hot, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('Number parameters of the model: ' + str(shared.count_params(tf.trainable_variables()))+'\n')

    # tensorflow: define optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # needed for batchnorm
    with tf.control_dependencies(update_ops):
        lr = tf.placeholder(tf.float32)
        if config['optimizer'] == 'SGD_clip':
            optimizer = tf.train.GradientDescentOptimizer(lr)
            gradients, variables = zip(*optimizer.compute_gradients(ce_loss))
            gradients_clip, _ = tf.clip_by_global_norm(gradients, 5.0)
            train_step = optimizer.apply_gradients(zip(gradients_clip, variables))
        elif config['optimizer'] == 'many_lr_audioset':
            MULTIPLY_LR = 0.0001 # used to have a different learning rate for my_model variables, i.e.: 0.0001
            print('Different learning rates: '+str(MULTIPLY_LR))
            opt = tf.train.GradientDescentOptimizer(lr)
            grads_and_vars = opt.compute_gradients(ce_loss)
            trainable = [gv for gv in grads_and_vars if 'my_model' in str(gv[1])]
            to_modify_lr = [gv for gv in grads_and_vars if 'my_model' not in str(gv[1])]
            for t in to_modify_lr: 
                trainable.append((t[0]*MULTIPLY_LR,t[1]))
            train_step = opt.apply_gradients(trainable)

    # save experimental settings
    experiment_id = 'fold_'+str(config_file.FOLD)+'_'+str(shared.get_epoch_time())
    experiment_folder = config_file.DATA_FOLDER + 'experiments/' + str(experiment_id) + '/'
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    config['fold'] = config_file.FOLD
    json.dump(config, open(experiment_folder + 'config.json', 'w'))
    print('\nConfig file saved: ' + str(config))

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
        fy.write('Update\tEpoch\tloss\tmean_distance_protos\taccuracy\n')
        fy.close()

        # saving a list of all the models trained for this experiment
        fall = open(experiment_folder + 'models.list', 'a')
        fall.write(str(model_id) +'\n')
        fall.close()

        # few-shot learning: ids selection
        tmp_ids = shared.few_shot_data_preparation(all_ids_train, all_ids_test, classes_vector, label2ids_train, label2ids_test, config)
        [ids_train, ids_test, label2selectedIDs] = tmp_ids

        # few-shot learning: fetch data
        tmp_data = proto.fetch_data(classes_vector, label2selectedIDs, id2audio_repr_path, id2gt_train, config, transfer_learning=True)
        [train_set_dic, train_gt_dic, train_id_dic, minimum_number_of_patches, total_number_of_patches] = tmp_data
               
        # tensorflow: create a session to run the tensorflow graph
        sess.run(tf.global_variables_initializer())

        # load AudioSet VGGish model
        vggish_vars = [v for v in tf.global_variables() if 'my_model' not in v.name]
        for variables in vggish_vars:
            print(variables)

        loader = tf.train.Saver(vggish_vars, name='vggish_load_pretrained', write_version=1)
        loader.restore(sess, 'vggish_model.ckpt')

        # training
        mean_distances = []
        epoch = 0
        perfect_epoch = 0
        update = 0
        acc_counter = 0
        define_support = True
        not_improve = 0
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

                # define query set, that changes every update
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
                print(np.asarray(support_set_ten).shape)

            # format for tensorflow: as np.array and channels last
            query_set = np.expand_dims(query_set_ten,axis=-1)
            query_gt = np.array(query_gt_ten)

            # training: iterations over the same data
            #import ipdb; ipdb.set_trace()
            [_, loss, protos_t, train_acc, mean_d, d_pr, d_pr_cos] = sess.run(
                                  [train_step, ce_loss, emb_prototypes, acc, mean_dists_protos, dists_protos, dists_protos_cos],
                                  feed_dict={x: support_set, q: query_set, y_one_hot: query_gt, lr: config['learning_rate']})

            print('%d/ epoch: %d | loss: %g | avg(D_protos): %g | # support: %d | # query: %d | acc: %g' 
                  % (update, epoch, loss, mean_d, support_set.shape[1], query_set.shape[0], train_acc))
            fy = open(model_folder + 'train_log.tsv', 'a')
            fy.write('%d\t%d\t%g\t%g\t%g\n' % (update, epoch, loss, mean_d, train_acc))
            fy.close()

            # stop criteria
            if update % (total_number_of_patches//query_set.shape[0]) == 0:
                epoch += 1

                # EVALUATING TRAIN SET # [id_string, save_latents, track_accuracies, printing, transfer learning?] 
                vis_vars = ['[epoch-eval]', False, True, True, True, model_folder]  
                tf_vars = [sess, x, q, log_p_y, emb_q, emb_prototypes]
                epoch_acc = proto.eval(config, ids_train, id2audio_repr_path, support_set, id2gt_train, id2label_train, tf_vars, vis_vars)

                if epoch_acc <= max_acc:
                    not_improve += 1
                    print('Did not improve! Max accuracy: '+str(max_acc))
                    if not_improve > config['max_accuracy']: # the same number of epochs without improving or having 100% accuracy
                        break
 
                if epoch_acc > max_acc:
                    max_acc = epoch_acc

                if  epoch > config['max_epochs']:
                    break


        # EVALUATING TRAIN SET AND STORE STUFF # [id_string, save_latents, track_accuracies, printing, transfer learning?] 
        vis_vars = ['train_set', True, False, True, True, model_folder]
        tf_vars = [sess, x, q, log_p_y, emb_q, emb_prototypes]
        train_acc = proto.eval(config, ids_train, id2audio_repr_path, support_set, id2gt_train, id2label_train, tf_vars, vis_vars)

        # EVALUATING TEST SET # [id_string, save_latents, track_accuracies, printing, transfer learning?]  
        vis_vars = ['test_set', True, False, True, True, model_folder]
        tf_vars = [sess, x, q, log_p_y, emb_q, emb_prototypes]
        accuracy = proto.eval(config, ids_test, id2audio_repr_path, support_set, id2gt_test, id2label_test, tf_vars, vis_vars)

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
    tr.close()
