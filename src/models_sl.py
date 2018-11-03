import tensorflow as tf

def model_number(x, is_training, config):

    if config['model_number'] == 0:
        print('\nMODEL: SB-CNN')
        print('-----------------------------------\n')
        return sb_cnn(x, is_training, config)

    elif config['model_number'] == 1:
        print('\nMODEL: SB-CNN | BN input')
        print('-----------------------------------\n')
        return sb_cnn_bn(x, is_training, config)

    elif config['model_number'] == 2:
        print('\nMODEL: Timbre | BN input')
        print('-----------------------------------\n')
        return timbre(x, is_training, config, num_filters=config['num_classes_dataset'])

    elif config['model_number'] == 3:
        print('\nMODEL: VGG | BN input')
        print('-----------------------------------\n')
        return vgg(x, is_training, config, num_filters=32)

    elif config['model_number'] == 11:
        print('\nMODEL: SB-CNN -> Justin | BN input | LOG learn')
        print('-----------------------------------\n')
        return sb_cnn_bn(log_learn(x), is_training, config)

    elif config['model_number'] == 12:
        print('\nMODEL: Timbre | MP -> direct | BN input | LOG learn')
        print('-----------------------------------\n')
        return timbre(log_learn(x), is_training, config, num_filters=config['num_classes_dataset'])

    elif config['model_number'] == 13:
        print('\nMODEL: VGG | BN input | LOG learn | 32 filters')
        print('-----------------------------------\n')
        return vgg(log_learn(x), is_training, config, num_filters=32)

    elif config['model_number'] == 14:
        print('\nMODEL: VGG | BN input | LOG learn | 128 filters')
        print('-----------------------------------\n')
        return vgg(log_learn(x), is_training, config, num_filters=128)

    raise RuntimeError("ERROR: Model {} can't be found!".format(config["model_number"]))


def log_learn(x):

    with tf.variable_scope('log_learn'):

        ta = tf.Variable(tf.constant(7, dtype=tf.float32), name='ta', trainable=True)
        ba = tf.Variable(tf.constant(1, dtype=tf.float32), name='ba', trainable=True)

        alpha = tf.exp(ta, name='alpha')
        beta = tf.log(1+tf.exp(ba), name='beta')

        return tf.log(tf.scalar_mul(alpha,x)+beta)


def vgg(x, is_training, config, num_filters):

    with tf.variable_scope('vggish'):
        
        print('[SMALL FILTERS] Input: ' + str(x.get_shape))
        input_layer = tf.expand_dims(x, 3)
        bn_input = tf.layers.batch_normalization(input_layer, training=is_training, axis=-1)
        conv1 = tf.layers.conv2d(inputs=bn_input,
                                 filters=num_filters,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='1CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
        bn_conv1 = tf.layers.batch_normalization(conv1, training=is_training, axis=-1)
        pool1 = tf.layers.max_pooling2d(inputs=bn_conv1, pool_size=[2, 2], strides=[2, 2])
        print(pool1.get_shape)

        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=num_filters,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='2CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
        bn_conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=-1)
        pool2 = tf.layers.max_pooling2d(inputs=bn_conv2, pool_size=[2, 2], strides=[2, 2])
        print(pool2.get_shape)

        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=num_filters,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='3CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
        bn_conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=-1)
        pool3 = tf.layers.max_pooling2d(inputs=bn_conv3, pool_size=[2, 2], strides=[2, 2])
        print(pool3.get_shape)

        conv4 = tf.layers.conv2d(inputs=pool3,
                                 filters=num_filters,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='4CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
        bn_conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=-1)
        pool4 = tf.layers.max_pooling2d(inputs=bn_conv4, pool_size=[2, 2], strides=[2, 2])
        print(pool4.get_shape)

        conv5 = tf.layers.conv2d(inputs=pool4, 
                                 filters=num_filters, 
                                 kernel_size=[3, 3], 
                                 padding='same', 
                                 activation=tf.nn.elu,
                                 name='5CNN', 
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
        bn_conv5 = tf.layers.batch_normalization(conv5, training=is_training, axis=-1)
        pool5 = tf.layers.max_pooling2d(inputs=bn_conv5, pool_size=[2, 2], strides=[2, 2])
        print(pool5.get_shape)

        flat = tf.layers.flatten(pool5)
        do = tf.layers.dropout(flat, rate=0.5, training=is_training)

        print(do.get_shape)
        output = tf.layers.dense(inputs=do,
                            activation=None,
                            units=config['num_classes_dataset'],
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))

    return output


def timbre(x, is_training, config, num_filters):

    with tf.variable_scope('timbre'):

        print('[CNN SINGLE] Input: ' + str(x.get_shape))

        input_layer = tf.expand_dims(x, 3)
        bn_input = tf.layers.batch_normalization(input_layer, training=is_training, axis=-1)
        conv1 = tf.layers.conv2d(inputs=bn_input,
                                 filters=num_filters,
                                 kernel_size=[7,108], # 7,86 / 7,108
                                 padding='valid',
                                 activation=None,
                                 name='1CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[conv1.shape[1],conv1.shape[2]], 
                                                      strides=[conv1.shape[1],conv1.shape[2]])
        output = tf.layers.flatten(pool1)
       
    print(conv1.get_shape)
    print(conv1.shape[1])
    print(conv1.shape[2])
    print(pool1.get_shape)
    print(output)

    return output


def sb_cnn_core(input_, is_training, config):

    print(input_.get_shape)
    conv1 = tf.layers.conv2d(inputs=input_,
                             filters=24,
                             kernel_size=[5, 5],
                             padding='valid',
                             activation=tf.nn.relu,
                             name='1CNN',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
    print(conv1.get_shape)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 2], strides=[4, 2])

    print(pool1.get_shape)
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=48,
                             kernel_size=[5, 5],
                             padding='valid',
                             activation=tf.nn.relu,
                             name='2CNN',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
    print(conv2.get_shape)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[4, 2], strides=[4, 2])

    print(pool2.get_shape)
    conv3 = tf.layers.conv2d(inputs=pool2,
                             filters=48,
                             kernel_size=[5, 5],
                             padding='valid',
                             activation=tf.nn.relu,
                             name='3CNN',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
    print(conv3.get_shape)
    flat_conv3 = tf.contrib.layers.flatten(conv3)

    print(flat_conv3.get_shape)
    do_pool5 = tf.layers.dropout(flat_conv3, rate=0.5, training=is_training)

    print(do_pool5.get_shape)
    dense_out = tf.layers.dense(inputs=do_pool5,
                            activation=tf.nn.relu,
                            units=64,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))


    do = tf.layers.dropout(dense_out, rate=0.5, training=is_training)

    print(do.get_shape)
    output = tf.layers.dense(inputs=do,
                            activation=None,
                            units=config['num_classes_dataset'],
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
    
    print('output: ' + str(output.get_shape))    
    return output


def sb_cnn(x, is_training, config):

    print('Input: ' + str(x.get_shape))
    input_layer = tf.expand_dims(x, 3)
    return sb_cnn_core(input_layer, is_training, config)


def sb_cnn_bn(x, is_training, config):

    print('Input: ' + str(x.get_shape))
    
    input_layer = tf.expand_dims(x, 3)
    print(input_layer.get_shape)
    bn_input = tf.layers.batch_normalization(input_layer, training=is_training, axis=-1)

    return sb_cnn_core(bn_input, is_training, config)

