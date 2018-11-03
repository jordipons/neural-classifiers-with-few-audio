import tensorflow as tf

def select(x, config, is_training, reuse=False):

    if config['model_number'] == 2:
        return vgg_bn(x, config, is_training, 10, reuse)

    elif config['model_number'] == 12:
        return vgg_bn(log_learn(x), config, is_training, 10, reuse)

    raise RuntimeError("ERROR: Model {} can't be found!".format(config["model_number"]))


def vgg_bn(x, config, is_training, output_filters, reuse=False):

    with tf.variable_scope('vggish', reuse=reuse):

        NUMBER_FILTERS = 128

        print('VGG with batchnorm! #filters: '+str(NUMBER_FILTERS))
        
        print('Input: ' + str(x.get_shape))
        bn_input = tf.layers.batch_normalization(x, training=is_training, axis=1)
        conv1 = tf.layers.conv2d(inputs=bn_input,
                                 filters=NUMBER_FILTERS,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='1CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
        bn_conv1 = tf.layers.batch_normalization(conv1, training=is_training, axis=-1)
        pool1 = tf.layers.max_pooling2d(inputs=bn_conv1, pool_size=[2, 2], strides=[2, 2])
        print(pool1.get_shape)

        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=NUMBER_FILTERS,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='2CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
        bn_conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=-1)
        pool2 = tf.layers.max_pooling2d(inputs=bn_conv2, pool_size=[2, 2], strides=[2, 2])
        print(pool2.get_shape)

        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=NUMBER_FILTERS,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='3CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
        bn_conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=-1)
        pool3 = tf.layers.max_pooling2d(inputs=bn_conv3, pool_size=[2, 2], strides=[2, 2])
        print(pool3.get_shape)

        conv4 = tf.layers.conv2d(inputs=pool3,
                                 filters=NUMBER_FILTERS,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='4CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
        bn_conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=-1)
        pool4 = tf.layers.max_pooling2d(inputs=bn_conv4, pool_size=[2, 2], strides=[2, 2])
        print(pool4.get_shape)

        conv5 = tf.layers.conv2d(inputs=pool4, 
                                 filters=NUMBER_FILTERS, 
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
                            units=output_filters,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))

        config['embedding_size'] = output.get_shape().as_list()[1]

        return [output, config]

def log_learn(x):

    with tf.variable_scope('log_learn'):

        ta = tf.Variable(tf.constant(7, dtype=tf.float32), name='ta', trainable=True)
        ba = tf.Variable(tf.constant(1, dtype=tf.float32), name='ba', trainable=True)

        alpha = tf.exp(ta, name='alpha')
        beta = tf.log(1+tf.exp(ba), name='beta')

        return tf.log(tf.scalar_mul(alpha,x)+beta)

