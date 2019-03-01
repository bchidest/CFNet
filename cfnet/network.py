import tensorflow as tf
import numpy as np
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
import cnn_params
from keras_gcnn.layers import GBatchNorm
from keras_gcnn.layers.pooling import GroupPool


MOVING_AVERAGE_DECAY = 0.98
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001


def weight_variable(shape, name, initial=None, std=None):
    ''' Create weight variable for a layer.
        Initialize variables and add l2 weight decay'''
    if std is None:
        stddev = 0.1
    else:
        stddev = std
    if initial is None:
        weights = tf.get_variable(name, shape,
                                  initializer=tf.random_normal_initializer(
                                      stddev=stddev),
                                  regularizer=tf.contrib.layers.l2_regularizer(
                                      scale=0.0005))
    else:
        weights = tf.get_variable(name, shape,
                                  initializer=tf.constant_initializer(initial),
                                  regularizer=tf.contrib.layers.l2_regularizer(
                                      scale=0.0005))
    return weights


def weight_variable_with_rotations(shape, n_rotations, name):
    '''Creates weight variable and then applies rotatations.
       NOTE: This is used for DFT transition layer, not convolutional
         layers. For convolutional layers, rotations are applied in
         rotational_conv2d functions.'''
    # Choose interpolation method ('BILINEAR' or 'NEAREST').
    # interpolation = 'NEAREST'
    interpolation = 'BILINEAR'
    rot_angle = 2*np.pi / n_rotations
    weights = weight_variable([shape[3], shape[0], shape[1], shape[2]], name)
    weights_rotated = []
    for r in range(1, n_rotations):
        weights_rotated.append(
                tf.contrib.image.rotate(weights, r*rot_angle,
                                        interpolation=interpolation))
    weights = tf.concat([weights] + weights_rotated, axis=0)
    weights = tf.transpose(weights, [1, 2, 3, 0])
    return weights


def bias_variable(shape, name, initial=None):
    '''Creates bias variable for a layer.'''
    if initial is None:
        bias = tf.get_variable(name, shape,
                               initializer=tf.constant_initializer(0.0))
    else:
        bias = tf.get_variable(name,
                               initializer=tf.convert_to_tensor(initial))
    return bias


def variable_summaries(var, name):
    '''Save summary statistics of variables of a layer.'''
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)


def conv2d(x, W, s, padding='VALID'):
    '''Simple wrapper around TF convolution function.'''
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding=padding)


def conic_conv2d(x, W, x_shape, W_shape, n_rotations, stride, padding='VALID'):
    interpolation = 'NEAREST'
    assert(W_shape[0] % 2 == 1)  # check that filter size is odd
    W_radius = W_shape[0] / 2  # radius excludes center pixel
    x_s = x_shape[1]  # could be odd or even
    mid = x_s / 2
    mid2 = (x_s - 2*W_radius) / 2

    assert(n_rotations in [4, 8])
    if n_rotations == 4:
        # To rotate W
        W_rot = []
        W_3D = tf.reshape(W, W_shape[:2] + [-1])    # Required by tf.image.rot90
        W_rot.append(W)
        W_rot.append(tf.reshape(tf.image.rot90(W_3D, -1), W_shape))
        W_rot.append(tf.reshape(tf.image.rot90(W_3D, -2), W_shape))
        W_rot.append(tf.reshape(tf.image.rot90(W_3D, -3), W_shape))

        # Calculate convolution on the `quarter' only
        tmp = [
            tf.nn.conv2d(x[:, :-mid+W_radius , :-mid+W_radius , :], W_rot[0], strides=[1,]*4, padding='VALID'),
            tf.nn.conv2d(x[:, :-mid+W_radius ,   mid-W_radius:, :], W_rot[1], strides=[1,]*4, padding='VALID'),
            tf.nn.conv2d(x[:,   mid-W_radius:,   mid-W_radius:, :], W_rot[2], strides=[1,]*4, padding='VALID'),
            tf.nn.conv2d(x[:,   mid-W_radius:, :-mid+W_radius , :], W_rot[3], strides=[1,]*4, padding='VALID'),
        ]

        # Merge the conv into the desired result
        if x_s % 2 == 0:
            x_out = [[
                tmp[0],
                tmp[1],
            ], [
                tmp[3],
                tmp[2],
            ]]
        else:
            x_out = [[
                tmp[0][:,   :-1,   :-1, :],
                tmp[1][:,   :-1,   : 1, :],
                tmp[1][:,   :-1,  1:  , :],
            ], [
                tmp[0][:, -1:  ,   :-1, :],
                (tmp[0][:, -1:, -1:, :] + tmp[1][:, -1:, :1, :] + tmp[2][:, :1, :1, :] + tmp[0][:, :1, -1:, :])/4,
                tmp[2][:,   :1 ,  1:  , :],
            ], [
                tmp[3][:,  1:  ,   :-1, :],
                tmp[3][:,  1:  , -1:  , :],
                tmp[2][:,  1:  ,  1:  , :],
            ]]
        x_out = [tf.concat(_, axis=2) for _ in x_out]
        x_out = tf.concat(x_out, axis=1)
        x_out = tf.squeeze(x_out)
    elif n_rotations == 8:
        # rotating X once and W 3+1 times is more expensive than rotating W 6+1 times
        W_rot = []
        W_3D = tf.reshape(W, W_shape[:2] + [-1])
        W_45 = tf.contrib.image.rotate(W_3D, np.pi/4, interpolation=interpolation)
        W_rot.append(W)
        W_rot.append(tf.reshape(tf.image.rot90(W_45, -1), W_shape))
        W_rot.append(tf.reshape(tf.image.rot90(W_3D, -1), W_shape))
        W_rot.append(tf.reshape(tf.image.rot90(W_45, -2), W_shape))
        W_rot.append(tf.reshape(tf.image.rot90(W_3D, -2), W_shape))
        W_rot.append(tf.reshape(tf.image.rot90(W_45, -3), W_shape))
        W_rot.append(tf.reshape(tf.image.rot90(W_3D, -3), W_shape))
        W_rot.append(tf.reshape(W_45, W_shape))
        #  \ 0 | 1 /
        # 7 \  |  / 2
        #    \ | /
        # -----------
        #    / | \
        # 6 /  |  \ 3
        #  / 5 | 4 \
        del W_3D, W_45

        # Calculate convolution on the `quarter' only
        tmp = [
            tf.nn.conv2d(x[:, :-mid+W_radius , :-mid+W_radius , :], W_rot[0], strides=[1,]*4, padding='VALID'),
            tf.nn.conv2d(x[:, :-mid+W_radius ,   mid-W_radius:, :], W_rot[1], strides=[1,]*4, padding='VALID'),
            tf.nn.conv2d(x[:, :-mid+W_radius ,   mid-W_radius:, :], W_rot[2], strides=[1,]*4, padding='VALID'),
            tf.nn.conv2d(x[:,   mid-W_radius:,   mid-W_radius:, :], W_rot[3], strides=[1,]*4, padding='VALID'),
            tf.nn.conv2d(x[:,   mid-W_radius:,   mid-W_radius:, :], W_rot[4], strides=[1,]*4, padding='VALID'),
            tf.nn.conv2d(x[:,   mid-W_radius:, :-mid+W_radius , :], W_rot[5], strides=[1,]*4, padding='VALID'),
            tf.nn.conv2d(x[:,   mid-W_radius:, :-mid+W_radius , :], W_rot[6], strides=[1,]*4, padding='VALID'),
            tf.nn.conv2d(x[:, :-mid+W_radius , :-mid+W_radius , :], W_rot[7], strides=[1,]*4, padding='VALID'),
        ]

        # We need masks to merge two triangular parts in the same `quarter'
        mask_np = np.zeros([mid2, ]*2, dtype=np.float32)
        mask_np[np.tril_indices(mid2, -1)] = 1
        mask_np[np.diag_indices(mid2    )] = .5
        maskll = tf.constant(mask_np[::  , ::  ][None, :, :, None])  # lower-left
        maskul = tf.constant(mask_np[::-1, ::  ][None, :, :, None])  # upper-left
        maskur = tf.constant(mask_np[::-1, ::-1][None, :, :, None])  # upper-right
        masklr = tf.constant(mask_np[::  , ::-1][None, :, :, None])  # lower-right
        # Merge the conv into the desired result
        if x_s % 2 == 0:
            x_out = [[
                tmp[7]*maskll + tmp[0]*maskur,
                tmp[1]*maskul + tmp[2]*masklr,
            ], [
                tmp[5]*masklr + tmp[6]*maskul,
                tmp[3]*maskur + tmp[4]*maskll,
            ]]
        else:
            x_out = [[
                tmp[7][:, :-1, :-1, :]*maskll + tmp[0][:, :-1, :-1, :]*maskur,
                (tmp[0][:, :-1, -1:, :] + tmp[1][:, :-1, :1, :]) / 2,
                tmp[1][:, :-1, 1:, :]*maskul + tmp[2][:, :-1, 1:, :]*masklr,
            ], [
                (tmp[6][:, :1, :-1, :] + tmp[7][:, -1:, :-1, :]) / 2,
                sum([
                    tmp[0][:, -1:, -1:, :],
                    tmp[1][:, -1:, :1 , :],
                    tmp[2][:, -1:, :1 , :],
                    tmp[3][:,  :1, :1 , :],
                    tmp[4][:,  :1, :1 , :],
                    tmp[5][:,  :1, -1:, :],
                    tmp[6][:,  :1, -1:, :],
                    tmp[7][:, -1:, -1:, :],
                ], 0) / 8,
                (tmp[2][:, -1:, 1:, :] + tmp[3][:, :1, 1:, :]) / 2,
            ], [
                tmp[5][:, 1:, :-1, :]*masklr + tmp[6][:, 1:, :-1, :]*maskul,
                (tmp[4][:, 1:, :1, :] + tmp[5][:, 1:, -1:, :]) / 2,
                tmp[3][:, 1:, 1:, :]*maskur + tmp[4][:, 1:, 1:, :]*maskll,
            ]]
        x_out = [tf.concat(_, axis=2) for _ in x_out]
        x_out = tf.concat(x_out, axis=1)
        x_out = tf.squeeze(x_out)
    else:
        assert False

    return x_out


class GBatchNorm_TF():

    def __init__(self):
        pass

    def make_var(self, name, shape, trainable_param=True, initializer_param=tf.keras.initializers.he_normal()):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=trainable_param, initializer=initializer_param)

    def run(self, x, x_size, x_depth, group_input, is_training):
        x_shape = [1, x_size, x_size, x_depth]
        params_shape = x_shape[-1:]

        axis = list(range(len(x_shape) - 1))

        one_init = tf.constant_initializer(value=1.0)
        zero_init = tf.constant_initializer(value=0.0)

        beta = self.make_var('beta', params_shape, initializer_param=zero_init)
        gamma = self.make_var('gamma', params_shape, initializer_param=one_init)
        moving_mean = self.make_var('moving_mean', params_shape,
                                    trainable_param=False, initializer_param=zero_init)
        moving_variance = self.make_var('moving_variance', params_shape,
                                        trainable_param=False, initializer_param=one_init)

        if is_training:
            mean, variance = tf.nn.moments(input, axis)
            if group_input != 'Z2':
                if group_input == 'C4':
                    num_repeat = 4
                else:
                    num_repeat = 8
                mean = tf.reshape(mean, [-1, num_repeat])
                mean = tf.reduce_mean(mean, 1, keep_dims=False)
                mean = tf.reshape(tf.tile(tf.expand_dims(mean, -1), [1, num_repeat]), [-1])
                variance = tf.reshape(variance, [-1, num_repeat])
                variance = tf.reduce_mean(variance, 1, keep_dims=False)
                variance = tf.reshape(tf.tile(tf.expand_dims(variance, -1), [1, num_repeat]), [-1])

            # update_moving_mean = moving_averages.assign_moving_average(moving_mean,
            #                                                            mean, BN_DECAY)
            # update_moving_variance = moving_averages.assign_moving_average(
            #     moving_variance, variance, BN_DECAY)
            # tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
            # tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

            train_mean = tf.assign(moving_mean,
                                   moving_mean * BN_DECAY + mean * (1 - BN_DECAY))
            train_var = tf.assign(moving_variance,
                                  moving_variance * BN_DECAY + variance * (1 - BN_DECAY))

            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(input,
                                                 mean, variance, beta, gamma, BN_EPSILON)
        else:
            return tf.nn.batch_normalization(input, moving_mean, moving_variance, beta, gamma, BN_EPSILON)


def conic_conv_layer(x, x_size, x_depth, layer_params, is_training,
                     activation=True, layer_name='conic_conv'):
    w_shape = [layer_params.filter_size,
               layer_params.filter_size,
               x_depth,
               layer_params.n_filters]
    x_shape = [1, x_size, x_size, x_depth]
    biases = bias_variable([layer_params.n_filters],
                           layer_name + '/biases')
    variable_summaries(biases, layer_name + '/biases')
    weights = weight_variable(w_shape,
                              layer_name + '/weights')
    variable_summaries(weights, layer_name + '/weights')
    h = conic_conv2d(x, weights, x_shape, w_shape,
                     layer_params.n_rotations,
                     layer_params.convolution_stride)\
        + biases
    # Batch normalization
    if layer_params.batch_norm:
        with tf.variable_scope(layer_name + '_bn'):
            h = tf.contrib.layers.batch_norm(
                        h, decay=0.95, center=True, scale=True,
                        is_training=is_training)
    # Pooling
    if layer_params.pooling:
        # Preserving rotation equivariance requires handling odd and even
        # sized inputs differently when pooling
        if x_size % 2 == 0:
            pool_sup = 2
        else:
            pool_sup = 3
        if layer_params.pooling == 'max':
            h = tf.nn.max_pool(h, [1, pool_sup, pool_sup, 1],
                               [1, layer_params.pooling_stride,
                                layer_params.pooling_stride, 1],
                               padding='VALID')
        elif layer_params.pooling == 'avg':
            h = tf.nn.avg_pool(h, [1, pool_sup, pool_sup, 1],
                               [1, layer_params.pooling_stride,
                                layer_params.pooling_stride, 1],
                               padding='VALID')
    # Activation
    if activation:
        h = tf.nn.relu(h)
    h_size = layer_params.calculate_output_size(x_size)
    return h, h_size


def g_conv_layer(x, x_size, x_depth, group_input, layer_params,
                 activation=True, layer_name='g-conv'):
    group_output = layer_params.ge_type
    gconv_indices, gconv_shape_info, w_shape =\
        gconv2d_util(h_input=group_input, h_output=group_output,
                     in_channels=x_depth,
                     out_channels=layer_params.n_filters,
                     ksize=layer_params.filter_size)
    weights = weight_variable(w_shape, layer_name + '/weights')
    variable_summaries(weights, layer_name + '/weights')
    h = gconv2d(input=x, filter=weights,
                strides=[1, layer_params.convolution_stride,
                         layer_params.convolution_stride, 1],
                padding=layer_params.padding,
                gconv_indices=gconv_indices,
                gconv_shape_info=gconv_shape_info)
    # Batch normalization
    if layer_params.batch_norm:
        with tf.variable_scope(layer_name + '_bn'):
            h = GBatchNorm(layer_params.ge_type)(h)
    # Pooling
    if layer_params.pooling:
        pool_sup = layer_params.pooling_support
        if layer_params.pooling == 'max':
            h = tf.nn.max_pool(h, [1, pool_sup, pool_sup, 1],
                               [1, layer_params.pooling_stride,
                                layer_params.pooling_stride, 1],
                               padding='VALID')
        elif layer_params.pooling == 'avg':
            h = tf.nn.avg_pool(h, [1, pool_sup, pool_sup, 1],
                               [1, layer_params.pooling_stride,
                                layer_params.pooling_stride, 1],
                               padding='VALID')
    # Activation
    if activation:
        h = tf.nn.relu(h)
    # G-Pooling
    if layer_params.ge_pool:
        h = GroupPool(layer_params.ge_type)(h)
    h_size = layer_params.calculate_output_size(x_size)
    return h, h_size


def conv_layer(x, x_size, x_depth, layer_params, is_training,
               activation=True, layer_name='conv'):
    biases = bias_variable([layer_params.n_filters],
                           layer_name + '/biases')
    variable_summaries(biases, layer_name + '/biases')
    weights = weight_variable([layer_params.filter_size,
                               layer_params.filter_size,
                               x_depth,
                               layer_params.n_filters],
                              layer_name + '/weights')
    variable_summaries(weights, layer_name + '/weights')

    h = conv2d(x, weights, layer_params.convolution_stride,
               padding=layer_params.padding) + biases
    # Batch normalization
    if layer_params.batch_norm:
        with tf.variable_scope(layer_name + '_bn'):
            h = tf.contrib.layers.batch_norm(
                    h, decay=0.95, center=True, scale=True,
                    is_training=is_training)
    # Pooling
    if layer_params.pooling:
        pool_sup = layer_params.pooling_support
        if layer_params.pooling == 'max':
            h = tf.nn.max_pool(h, [1, pool_sup, pool_sup, 1],
                               [1, layer_params.pooling_stride,
                                layer_params.pooling_stride, 1],
                               padding='VALID')
        elif layer_params.pooling == 'avg':
            h = tf.nn.avg_pool(h, [1, pool_sup, pool_sup, 1],
                               [1, layer_params.pooling_stride,
                                layer_params.pooling_stride, 1],
                               padding='VALID')
    # Activation
    if activation:
        h = tf.nn.relu(h)
    h_size = layer_params.calculate_output_size(x_size)
    return h, h_size


def fc_layer(x, x_n_nodes, layer_params, is_training, activation=True, layer_name='fc'):
    weights = weight_variable([x_n_nodes, layer_params.n_nodes],
                              layer_name + '/weights')
    variable_summaries(weights, layer_name + '/weights')
    bias = bias_variable([layer_params.n_nodes],
                         layer_name + '/biases')
    variable_summaries(bias, layer_name + '/biases')
    # Pass through weights
    h = tf.matmul(x, weights) + bias
    # Batch normalization
    if layer_params.batch_norm:
        with tf.variable_scope(layer_name + '_bn'):
            h = tf.contrib.layers.batch_norm(h, decay=0.95,
                                             center=True, scale=True,
                                             is_training=is_training)
    # Activation
    if activation:
        h = tf.nn.relu(h)
    return h


def dft_transition(x, x_n_nodes, n_rotations):
    # Cast to complex
    x = tf.cast(x, tf.complex64)
    x = tf.fft2d(x)
    x = tf.abs(x)
#    mask_abs = np.ones((batch_size, n_nodes, n_rotations), dtype='bool')
#    mask_abs[:, 0, 0] = 0
#    mask_abs = tf.constant(mask_abs)
#    layer_fft_abs = tf.abs(layer_fft)
#    layer_fft_real = tf.real(layer_fft)
#    layer_fft = tf.where(mask_abs, layer_fft_abs, layer_fft_real)
    # Normalize DFT output
    x = x / (x_n_nodes*n_rotations)
    return x


def split_g_conv_rotations(x, x_size, x_depth):
    x = tf.transpose(x, [1, 2, 3, 0])
    x = tf.reshape(x, [x_size**2 * x_depth, 4, -1])
    x = tf.transpose(x, [2, 0, 1])
    return x


def dft_transition_with_rotations(x, x_size, x_depth, layer_params,
                                  layer_name='conv-to-fc'):
    with tf.name_scope(layer_name):
        # Inner product of input feature map with rotated weights
        weights = \
            weight_variable_with_rotations([x_size,
                                            x_size,
                                            x_depth,
                                            layer_params.n_nodes],
                                           layer_params.n_rotations,
                                           layer_name + '/weights')
        variable_summaries(weights, layer_name + '/weights')
        biases = bias_variable([layer_params.n_nodes],
                               layer_name + '/biases')
        variable_summaries(biases, layer_name + '/biases')
        biases = tf.tile(biases, [layer_params.n_rotations])
        h = conv2d(x, weights, 1) + biases
        # Activation
        h = tf.nn.relu(h)
        # Output should be [batch_size, 1, 1, n_nodes*n_rotations], so squeeze
        h = tf.squeeze(h)
        # Reshape to be [batch_size, n_nodes, n_rotations]
        h = tf.transpose(h, [1, 0])
        h = tf.reshape(h, [layer_params.n_rotations,
                           layer_params.n_nodes, -1])
        h = tf.transpose(h, [2, 1, 0])
        # DFT
        h = dft_transition(h, layer_params.n_nodes,
                           layer_params.n_rotations)
    return h


def inference(x, params, input_size, input_depth, batch_size, is_training,
              keep_prob, verbose=False):
    h = x
    h_size = input_size
    h_depth = input_depth

    n_conv_layers = len(params.conv_layers)
    n_fc_layers = len(params.fully_connected_layers)

    # Convolutional layers
    for i, layer_params in zip(range(n_conv_layers), params.conv_layers):
        activation = True
        layer_name = 'conv' + str(i)
        if verbose:
            print(layer_name + ': ' + str(layer_params))
        with tf.name_scope(layer_name):
            # Activation
            if i == n_conv_layers - 1:
                activation = False
            # Conic Convolution
            if isinstance(layer_params, cnn_params.REConvolutionalLayer):
                h, h_size = conic_conv_layer(h, h_size, h_depth, layer_params,
                                             is_training, activation, layer_name)
            # Group-Equivariant Convolution
            elif isinstance(layer_params, cnn_params.GEConvolutionalLayer):
                # Determine group of input
                if i == 0:
                    group_input = 'Z2'
                elif not isinstance(params.conv_layers[i - 1], cnn_params.GEConvolutionalLayer):
                    group_input = 'Z2'
                else:
                    group_input = params.conv_layers[i - 1].ge_type
                h, h_size = g_conv_layer(h, h_size, h_depth, group_input,
                                         layer_params, activation, layer_name)
            # Standard Convolution
            else:
                h, h_size = conv_layer(h, h_size, h_depth, layer_params,
                                       is_training, activation, layer_name)
            # Dropout
            h = tf.nn.dropout(h, keep_prob)
            h_depth = layer_params.n_filters
            tf.summary.histogram(layer_name + '/activations', h)
            if verbose:
                print('\t output feature map height, width = %d\n' % h_size)

    # Transition from convolutional to fully-connected layers via DFT
    layer_params = params.transition_layer
    if layer_params:
        layer_name = 'conv-to-fc'
        if verbose:
            print(layer_name + ': ' + str(layer_params))
        # If conic convolution
        if isinstance(params.conv_layers[-1], cnn_params.REConvolutionalLayer):
            h = tf.nn.relu(h)
            # Calculate the width/height of the last conic conv feature map
            h = dft_transition_with_rotations(h, h_size,
                                              params.conv_layers[-1].n_filters,
                                              layer_params, layer_name)
            h_n_nodes = layer_params.n_nodes*layer_params.n_rotations
        # Else if G-convolution
        elif isinstance(params.conv_layers[-1], cnn_params.GEConvolutionalLayer):
            # Cannot perform DFT transition if G-pooling was applied
            assert(not params.conv_layers[-1].ge_pool)
            # Cannot perform DFT transition if group is D4
            assert(params.conv_layers[-1].ge_type != 'D4')
            # Spatial dimension must have been removed through conv layers
            assert(h_size == 1)
            # Reshape output to be [batch_size, h_size, h_size, n_filters, 4]
            h = split_g_conv_rotations(h, h_size, params.conv_layers[-1].n_filters,
                                       params.conv_layers[-1])
            h = dft_transition(h, layer_params.n_nodes,
                               layer_params.n_rotations)
            h_n_nodes = params.conv_layers[-1].n_filters*layer_params.n_rotations
    else:
        h_n_nodes = h_size**2*params.conv_layers[-1].n_filters
        if isinstance(params.conv_layers[-1], cnn_params.GEConvolutionalLayer) \
                and not params.conv_layers[-1].ge_pool:
            # Need to compute final G-conv feature map dimensions
            if params.conv_layers[-1].ge_type == 'C4':
                h_n_nodes = h_n_nodes*4
            elif params.conv_layers[-1].ge_type == 'D4':
                h_n_nodes = h_n_nodes*8
        # If fully-connected layers, apply activation to conv feature map
        if n_fc_layers > 0:
            h = tf.nn.relu(h)

    # Reshape before fully connected layers
    h = tf.reshape(h, [batch_size, -1])
    # If transition layer, then apply batch norm after reshape
    if params.transition_layer:
        with tf.variable_scope(layer_name + '2' + '_bn'):
            h = tf.contrib.layers.batch_norm(
                    h, decay=0.95, center=True, scale=True,
                    is_training=is_training)

    # Fully-connected layers
    for i, layer_params in zip(range(n_fc_layers), params.fully_connected_layers):
        activation = True
        layer_name = 'fc' + str(i)
        if verbose:
            print(layer_name + ': ' + str(layer_params))
        if i == n_fc_layers - 1:
            activation = False
        with tf.name_scope(layer_name):
            h = fc_layer(h, h_n_nodes, layer_params, is_training, activation,
                         layer_name)
            tf.summary.histogram(layer_name + '/activations', h)
            h_n_nodes = layer_params.n_nodes
    return h


def loss(logits, labels, one_hot=False):
    if one_hot:
        labels = tf.cast(labels, tf.float32)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
    else:
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    prediction_loss = cross_entropy_mean
    tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
    return prediction_loss


def training(loss, global_step, learning_rate, decay_step):
    # Decay learning rate
    lr = tf.train.exponential_decay(learning_rate, global_step, decay_step, 0.95, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    # Other optimizers could be inserted here
    train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
    #train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
    return train_step


def evaluation(logits, labels, one_hot):
    if one_hot:
        correct = tf.nn.in_top_k(logits, tf.argmax(labels, 1), 1)
    else:
        correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
