import tensorflow as tf
import math
import numpy as np
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
import cnn_params
from keras_gcnn.layers import GBatchNorm
from keras_gcnn.layers.pooling import GroupPool


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
    interpolation = 'NEAREST'
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


def variable_summaries(var, name, visualize=False):
    '''Save summary statistics of variables of a layer.'''
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)
    if visualize:
        var2 = tf.transpose(var, perm=[3, 0, 1, 2])
        for i in range(var2.get_shape()[-1]):
            tf.summary.image(name="first_layer_filters_" + str(i),
                             tensor=tf.expand_dims(var2[..., i], -1),
                             max_outputs=var2.get_shape()[0])


def groupy_pool_group(x, shape, h_input):
    # Pools over filter dimension. Every eight contiguous filters
    # correspond to one group.
    if h_input == 'D4':
        nto = 8
    elif h_input == 'C4':
        nto = 4
    temp = tf.reshape(x, shape + [nto])
    temp = tf.reduce_max(temp, reduction_indices=[4])
    return tf.reshape(temp, shape)


def visualize_conv_feature_map(var, name):
    feat_map_summaries = []
    for i in range(var.get_shape()[-1]):
        feat_map_summaries.append(tf.expand_dims(var[..., i], -1))
    return feat_map_summaries


def conv2d(x, W, s, padding='VALID'):
    '''Simple wrapper around TF convolution function.'''
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding=padding)


def rotation_equivariant_conv2d_simple(x, W, x_shape, W_shape, n_rotations, stride,
                                       padding='VALID'):
    interpolation = 'NEAREST'
    batch_size = x_shape[0]
    W_s = W_shape[0]  # filter will always be odd
    W_radius = W_s / 2  # radius will therefore exclude center pixel
    W_d = W_shape[3]
    x_s = x_shape[1]  # could be odd or even
    mid = x_s / 2
    x_s_new = x_s - 2*W_radius
    mid2 = x_s_new / 2

    assert(n_rotations in [4, 8])
    if n_rotations == 4:
        W = tf.expand_dims(W, 0)
        x_rot = tf.expand_dims(x, 1)
        x_rot = tf.concat([x_rot, tf.expand_dims(tf.image.rot90(x, -1), 1)], axis=1)
        x_rot = tf.concat([x_rot, tf.expand_dims(tf.image.rot90(x, -2), 1)], axis=1)
        x_rot = tf.concat([x_rot, tf.expand_dims(tf.image.rot90(x, -3), 1)], axis=1)
        mask_np = np.zeros((1, 4, x_s_new, x_s_new, 1), dtype='float32')
        if x_s % 2 == 0:
            print('\t\t' + str(x_s) + ', 4 rotations, input size even')
#            mask_np[0, 0, 0:mid-1, 0:mid-1] = 1
#            mask_np[0, 1] = np.rot90(mask_np[0, 0], -1)
#            mask_np[0, 2] = np.rot90(mask_np[0, 0], -2)
#            mask_np[0, 3] = np.rot90(mask_np[0, 0], -3)
            mask_np[0, :, 0:mid2, 0:mid2] = 1
        else:
            print('\t\t' + str(x_s) + ', 4 rotations, input size odd')
#            mask_np[0, 0, 0:mid+1, 0:mid] = 1
#            mask_np[0, 1] = np.rot90(mask_np[0, 0], -1)
#            mask_np[0, 2] = np.rot90(mask_np[0, 0], -2)
#            mask_np[0, 3] = np.rot90(mask_np[0, 0], -3)
            mask_np[0, :, 0:mid2+1, 0:mid2] = 1
            mask_np[0, :, mid2, mid2] = 0.25
        mask = tf.convert_to_tensor(mask_np)

        x_out = tf.nn.conv3d(x_rot, W, strides=[1, 1, 1, 1, 1], padding='VALID')
        x_out = tf.multiply(x_out, mask)
#        x_out = x_out[:, 0] + tf.image.rot90(x_out[:, 1], 1) +\
#                    tf.image.rot90(x_out[:, 2], 2) + tf.image.rot90(x_out[:, 3], 3)
#        x_out = tf.squeeze(x_out)
        x_out = tf.squeeze(x_out[:, 0]) +\
            tf.squeeze(tf.image.rot90(x_out[:, 1], 1)) +\
            tf.squeeze(tf.image.rot90(x_out[:, 2], 2)) +\
            tf.squeeze(tf.image.rot90(x_out[:, 3], 3))
    elif n_rotations == 8:
        x_rot = tf.expand_dims(x, 1)
        x_rot = tf.concat([x_rot, tf.expand_dims(tf.image.rot90(x, -1), 1)], axis=1)
        x_rot = tf.concat([x_rot, tf.expand_dims(tf.image.rot90(x, -2), 1)], axis=1)
        x_rot = tf.concat([x_rot, tf.expand_dims(tf.image.rot90(x, -3), 1)], axis=1)
        mask_np_ud = np.zeros((1, 4, x_s_new, x_s_new, 1), dtype='float32')
        mask_np_ld = np.zeros((1, 4, x_s_new, x_s_new, 1), dtype='float32')
        if x_s % 2 == 0:
            print('\t\t' + str(x_s) + ', 8 rotations, input size even')
            for i in range(mid2):
                for j in range(mid2):
                    if j > i:
                        mask_np_ud[0, :, i, j] = 1
                    elif j == i:
                        mask_np_ld[0, :, i, j] = 0.5
                        mask_np_ud[0, :, i, j] = 0.5
                    else:
                        mask_np_ld[0, :, i, j] = 1
        else:
            print('\t\t' + str(x_s) + ', 8 rotations, input size odd')
            for i in range(mid2):
                for j in range(mid2):
                    if j > i:
                        mask_np_ud[0, :, i, j] = 1
                    elif j == i:
                        mask_np_ld[0, :, i, j] = 0.5
                        mask_np_ud[0, :, i, j] = 0.5
                    else:
                        mask_np_ld[0, :, i, j] = 1
            mask_np_ud[0, :, 0:mid2, mid2] = 0.5
            mask_np_ld[0, :, mid2, 0:mid2] = 0.5
            mask_np_ud[0, :, mid2, mid2] = 0.125
            mask_np_ld[0, :, mid2, mid2] = 0.125
        mask_ud = tf.convert_to_tensor(mask_np_ud)
        mask_ld = tf.convert_to_tensor(mask_np_ld)
#        mask = tf.convert_to_tensor(mask_np_ud + mask_np_ld)
        W_45 = tf.transpose(W, [3, 0, 1, 2])
        W_45 = tf.contrib.image.rotate(W_45, np.pi / 4, interpolation=interpolation)
        W_45 = tf.transpose(W_45, [1, 2, 3, 0])
        W = tf.expand_dims(W, 0)
        W_45 = tf.expand_dims(W_45, 0)

        x_out = tf.nn.conv3d(x_rot, W, strides=[1, 1, 1, 1, 1], padding='VALID')
        x_45_out = tf.nn.conv3d(x_rot, W_45, strides=[1, 1, 1, 1, 1], padding='VALID')
        x_out = tf.multiply(x_out, mask_ud)
        x_45_out = tf.multiply(x_45_out, mask_ld)
#        x_out = x_out[:, 0] + tf.image.rot90(x_out[:, 1], 1) +\
#                    tf.image.rot90(x_out[:, 2], 2) + tf.image.rot90(x_out[:, 3], 3) +\
#        x_out = tf.nn.conv3d(x_rot, W, strides=[1, 1, 1, 1, 1], padding='VALID')
#        x_out = tf.multiply(x_out, mask)
        x_out = tf.squeeze(x_out[:, 0]) +\
            tf.squeeze(tf.image.rot90(x_out[:, 1], 1)) +\
            tf.squeeze(tf.image.rot90(x_out[:, 2], 2)) +\
            tf.squeeze(tf.image.rot90(x_out[:, 3], 3)) +\
            tf.squeeze(x_45_out[:, 0]) +\
            tf.squeeze(tf.image.rot90(x_45_out[:, 1], 1)) +\
            tf.squeeze(tf.image.rot90(x_45_out[:, 2], 2)) +\
            tf.squeeze(tf.image.rot90(x_45_out[:, 3], 3))
        x_out = tf.squeeze(x_out)

    return x_out


def rotation_equivariant_conv2d_simple2(x, W, x_shape, W_shape, n_rotations, stride,
                                       padding='VALID'):
    # print('-----')
    interpolation = 'NEAREST'
    batch_size = x_shape[0]
    W_s = W_shape[0]  # filter will always be odd
    W_radius = W_s / 2  # radius will therefore exclude center pixel
    W_d = W_shape[3]
    x_s = x_shape[1]  # could be odd or even
    mid = x_s / 2
    x_s_new = x_s - 2*W_radius
    mid2 = x_s_new / 2

    # assert(n_rotations in [4, 8])
    if n_rotations == 4:
        W_rot = []
        W_3D = tf.reshape(W, W_shape[:2] + [-1])
        W_rot.append(W_3D)
        W_rot.append(tf.image.rot90(W_3D, -1))
        W_rot.append(tf.image.rot90(W_3D, -2))
        W_rot.append(tf.image.rot90(W_3D, -3))
        W_rot = tf.concat([tf.reshape(_, W_shape) for _ in W_rot], axis=3)

        # print(x.shape)
        # print(W_rot.shape)

        x_out = tf.nn.conv2d(x, W_rot, strides=[1, 1, 1, 1], padding='VALID')
        # print(x_out.shape)

        mask_np = np.zeros((x_s_new, x_s_new, 4), dtype=np.float32)
        if x_s % 2 == 0:
            print('\t\t' + str(x_s) + ', 4 rotations, input size even')
            mask_np[:mid2, :mid2, 0] = 1
            mask_np[:mid2, mid2:, 1] = 1
            mask_np[mid2:, mid2:, 2] = 1
            mask_np[mid2:, :mid2, 3] = 1
        else:
            print('\t\t' + str(x_s) + ', 4 rotations, input size odd')
            mask_np[:mid2  , :mid2  , 0] = 1
            mask_np[:mid2  , mid2+1:, 1] = 1
            mask_np[mid2+1:, mid2+1:, 2] = 1
            mask_np[mid2+1:, :mid2  , 3] = 1
            mask_np[mid2   , :mid2  , 0] = 1
            mask_np[:mid2  , mid2   , 1] = 1
            mask_np[mid2   , mid2+1:, 2] = 1
            mask_np[mid2+1:, mid2   , 3] = 1
            # mask_np[mid2   , :mid2  , [0, 3]] = .5
            # mask_np[:mid2  , mid2   , [0, 1]] = .5
            # mask_np[mid2   , mid2+1:, [1, 2]] = .5
            # mask_np[mid2+1:, mid2   , [2, 3]] = .5
            mask_np[mid2, mid2, :] = .25
        # print(mask_np.sum(axis=2))
        # print(mask_np[:, :, 0])
        # print(mask_np[:, :, 1])
        # print(mask_np[:, :, 2])
        # print(mask_np[:, :, 3])
#        mask = tf.convert_to_tensor(mask_np.reshape([1, x_s_new, x_s_new, 4, 1]))
        mask = tf.constant(mask_np.reshape([1, x_s_new, x_s_new, 4, 1]))

        x_out = tf.reshape(x_out, [batch_size, x_s_new, x_s_new, 4, W_d])
        x_out = tf.multiply(x_out, mask)
        # print(x_out.shape)
#        x_out = tf.math.reduce_sum(x_out, axis=3)
        x_out = tf.reduce_sum(x_out, axis=3)
        x_out = tf.squeeze(x_out)
    elif n_rotations == 8:
        # rotating X once and W 3+1 times is more expensive than rotating W 6+1 times
        W_rot = []
        W_3D = tf.reshape(W, W_shape[:2] + [-1])
        W_45 = tf.contrib.image.rotate(W_3D, np.pi/4, interpolation=interpolation)
        W_rot.append(W_45)
        W_rot.append(W_3D)
        W_rot.append(tf.image.rot90(W_45, -1))
        W_rot.append(tf.image.rot90(W_3D, -1))
        W_rot.append(tf.image.rot90(W_45, -2))
        W_rot.append(tf.image.rot90(W_3D, -2))
        W_rot.append(tf.image.rot90(W_45, -3))
        W_rot.append(tf.image.rot90(W_3D, -3))
        W_rot = tf.concat([tf.reshape(_, W_shape) for _ in W_rot], axis=3)
        del W_3D, W_45

        x_out = tf.nn.conv2d(x, W_rot, strides=[1, 1, 1, 1], padding='VALID')
        # print(x_out.shape)

        mask_np = np.zeros((x_s_new, x_s_new, 8), dtype=np.float32)
        if x_s % 2 == 0:
            print('\t\t' + str(x_s) + ', 8 rotations, input size even')
            # n = (mid2-1)*(mid2-2)/2
            # mask_np[np.triu_indices(mid2, 1) + ([0]*n)] = 1
            # mask_np[np.tril_indices(mid2, 1) + ([1]*n)] = 1
            mask_np[:mid2, :mid2, 0  ][np.tril_indices(mid2, -1)] = 1
            mask_np[:mid2, :mid2, 0:2][np.diag_indices(mid2,  2)] = .5
            mask_np[:mid2, :mid2, 1  ][np.triu_indices(mid2,  1)] = 1
            mask_np[:mid2, mid2:, 2  ][::-1][np.tril_indices(mid2, -1)] = 1
            mask_np[:mid2, mid2:, 2:4][::-1][np.diag_indices(mid2,  2)] = .5
            mask_np[:mid2, mid2:, 3  ][::-1][np.triu_indices(mid2,  1)] = 1
            mask_np[mid2:, mid2:, 4  ][np.triu_indices(mid2,  1)] = 1
            mask_np[mid2:, mid2:, 4:6][np.diag_indices(mid2,  2)] = .5
            mask_np[mid2:, mid2:, 5  ][np.tril_indices(mid2, -1)] = 1
            mask_np[mid2:, :mid2, 6  ][::-1][np.triu_indices(mid2,  1)] = 1
            mask_np[mid2:, :mid2, 6:8][::-1][np.diag_indices(mid2,  2)] = .5
            mask_np[mid2:, :mid2, 7  ][::-1][np.tril_indices(mid2, -1)] = 1
        else:
            print('\t\t' + str(x_s) + ', 8 rotations, input size odd')
            mask_np[:mid2  , :mid2  , 0  ][np.tril_indices(mid2, -1)] = 1
            mask_np[:mid2  , :mid2  , 0:2][np.diag_indices(mid2,  2)] = .5
            mask_np[:mid2  , :mid2  , 1  ][np.triu_indices(mid2,  1)] = 1
            mask_np[:mid2  , mid2+1:, 2  ][::-1][np.tril_indices(mid2, -1)] = 1
            mask_np[:mid2  , mid2+1:, 2:4][::-1][np.diag_indices(mid2,  2)] = .5
            mask_np[:mid2  , mid2+1:, 3  ][::-1][np.triu_indices(mid2,  1)] = 1
            mask_np[mid2+1:, mid2+1:, 4  ][np.triu_indices(mid2,  1)] = 1
            mask_np[mid2+1:, mid2+1:, 4:6][np.diag_indices(mid2,  2)] = .5
            mask_np[mid2+1:, mid2+1:, 5  ][np.tril_indices(mid2, -1)] = 1
            mask_np[mid2+1:, :mid2  , 6  ][::-1][np.triu_indices(mid2,  1)] = 1
            mask_np[mid2+1:, :mid2  , 6:8][::-1][np.diag_indices(mid2,  2)] = .5
            mask_np[mid2+1:, :mid2  , 7  ][::-1][np.tril_indices(mid2, -1)] = 1
            mask_np[mid2   , :mid2  , [0, 7]] = .5
            mask_np[:mid2  , mid2   , [1, 2]] = .5
            mask_np[mid2   , mid2+1:, [3, 4]] = .5
            mask_np[mid2+1:, mid2   , [5, 6]] = .5
            mask_np[mid2, mid2, :] = .125
        # for m in range(8):
        #     print(m)
        #     print(mask_np[:, :, m])
#        mask = tf.convert_to_tensor(mask_np.reshape([1, x_s_new, x_s_new, 8, 1]))
        mask = tf.constant(mask_np.reshape([1, x_s_new, x_s_new, 8, 1]))

        x_out = tf.reshape(x_out, [batch_size, x_s_new, x_s_new, 8, W_d])
        x_out = tf.multiply(x_out, mask)
#        x_out = tf.math.reduce_sum(x_out, axis=3)
        x_out = tf.reduce_sum(x_out, axis=3)
        x_out = tf.squeeze(x_out)
    else:
        assert False

    return x_out


def rotation_equivariant_conv2d_simple4(x, W, x_shape, W_shape, n_rotations, stride,
                                       padding='VALID'):
    interpolation = 'NEAREST'
    batch_size = x_shape[0]
    W_s = W_shape[0]  # filter will always be odd
    W_radius = W_s / 2  # radius will therefore exclude center pixel
    W_d = W_shape[3]
    x_s = x_shape[1]  # could be odd or even
    mid = x_s / 2
    x_s_new = x_s - 2*W_radius
    mid2 = x_s_new / 2

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


def dft2d_transition(layer, n_rotations, batch_size, conv_size,
                     n_filters_previous_layer, n_nodes):
    # Reshape output to be [batch_size, n_nodes, n_rotations]

#    layer = tf.reshape(layer, [n_nodes,
#                               n_rotations, batch_size])
#    layer = tf.transpose(layer, [2, 0, 1])
    layer = tf.transpose(layer, [1, 0])
    layer = tf.reshape(layer, [n_rotations,
                               n_nodes, batch_size])
    layer = tf.transpose(layer, [2, 1, 0])
    # DFT to enforce rotational invariance
    pre_activations = layer
    layer = tf.cast(layer, tf.complex64)
    layer_fft = tf.fft2d(layer)
    mask_abs = np.ones((batch_size, n_nodes, n_rotations), dtype='bool')
    mask_abs[:, 0, 0] = 0
#    mask_abs = tf.convert_to_tensor(mask_abs)
    mask_abs = tf.constant(mask_abs)
    layer_fft_abs = tf.abs(layer_fft)
    layer_fft_real = tf.real(layer_fft)
    layer_fft = tf.where(mask_abs, layer_fft_abs, layer_fft_real)
    # Normalize DFT output
    layer_fft = layer_fft / (conv_size*conv_size*n_filters_previous_layer*n_rotations)
    return layer_fft, pre_activations


def inference(x, params, input_size, input_depth, batch_size, n_classes,
              is_training, keep_prob, visualize_conv=False,
              visualize=False, radial_split=False):
    layer_in = x
    layer_ind = 0
    n_filters_previous_layer = input_depth
    layer_in_shape = [batch_size, input_size, input_size, input_depth]
    feat_map_summaries = []
    activations_to_viz = []
    weights_to_viz = []
    prev_layer = None
    n_conv_layers = len(params.conv_layers)
    n_fc_layers = len(params.fully_connected_layers)

    # Convolutional layers
    for i, layer in zip(range(n_conv_layers), params.conv_layers):
        layer_name = 'conv' + str(layer_ind)
        print(layer_name + ': ' + str(layer))
        with tf.name_scope(layer_name):
            biases = bias_variable([layer.n_filters],
                                   layer_name + '/biases')
            variable_summaries(biases, layer_name + '/biases')
            layer_in_size = layer.calculate_n_output_size(layer_in_shape[1])
            # RE-Convolution
            if isinstance(layer, cnn_params.REConvolutionalLayer):
                if layer_ind > 0 and isinstance(prev_layer, cnn_params.GEConvolutionalLayer):
                    if not prev_layer.ge_pool:
                        n_filters_previous_layer = n_filters_previous_layer * 4
                weights = weight_variable([layer.filter_size,
                                           layer.filter_size,
                                           n_filters_previous_layer,
                                           layer.n_filters],
                                          layer_name + '/weights')
                variable_summaries(weights, layer_name + '/weights')
#                temp = rotation_equivariant_conv2d(layer_in, weights,
                temp = rotation_equivariant_conv2d_simple4(layer_in, weights,
                                                   layer_in_shape,
                                                   [layer.filter_size,
                                                    layer.filter_size,
                                                    n_filters_previous_layer,
                                                    layer.n_filters],
                                                   layer.n_rotations,
                                                   layer.convolution_stride)\
                    + biases
            # GE-Convolution
            elif isinstance(layer, cnn_params.GEConvolutionalLayer):
                if layer_ind == 0:
                    h_input = 'Z2'
                    h_output = layer.ge_type
                elif not isinstance(params.conv_layers[layer_ind - 1], cnn_params.GEConvolutionalLayer):
                    h_input = 'Z2'
                    h_output = layer.ge_type
                else:
                    h_input = h_output
                    h_output = layer.ge_type
                print('G-CNN - ' + h_input + '-' + h_output)
                print(layer_in_size)
                gconv_indices, gconv_shape_info, w_shape =\
                    gconv2d_util(h_input=h_input, h_output=h_output,
                                 in_channels=n_filters_previous_layer,
                                 out_channels=layer.n_filters,
                                 ksize=layer.filter_size)
                weights = weight_variable(w_shape, layer_name + '/weights')
                variable_summaries(weights, layer_name + '/weights')
                temp = gconv2d(input=layer_in, filter=weights,
                               strides=[1, layer.convolution_stride,
                                        layer.convolution_stride, 1],
                               padding=layer.padding,
                               gconv_indices=gconv_indices,
                               gconv_shape_info=gconv_shape_info)
            # Standard Convolution
            else:
                weights = weight_variable([layer.filter_size,
                                           layer.filter_size,
                                           n_filters_previous_layer,
                                           layer.n_filters],
                                          layer_name + '/weights')
                variable_summaries(weights, layer_name + '/weights')
                temp = conv2d(layer_in, weights,
                              layer.convolution_stride,
                              padding=layer.padding) + biases
                print(layer_in_size)
            # Batch normalization
            if layer.batch_norm:
                with tf.variable_scope(layer_name + '_bn'):
                    if isinstance(layer, cnn_params.GEConvolutionalLayer):
                        temp = GBatchNorm(layer.ge_type)(temp)
                    else:
                        temp = tf.contrib.layers.batch_norm(
                                temp, decay=0.95, center=True, scale=True,
    #                            temp, center=True, scale=True,
                                is_training=is_training)
            # Pooling
            if layer.pooling:
                print(layer.pooling_support, layer.pooling_stride)
                if isinstance(layer, cnn_params.REConvolutionalLayer) or isinstance(layer, cnn_params.GEConvolutionalLayer):
                    x_s = layer_in_shape[1]
                    if x_s % 2 == 0:
                        pool_sup = 2
                    else:
                        pool_sup = 3
                else:
                    pool_sup = layer.pooling_support
                if layer.pooling == 'max':
                    temp = tf.nn.max_pool(temp,
                                          [1, pool_sup,
                                           pool_sup, 1],
                                          [1, layer.pooling_stride,
                                           layer.pooling_stride, 1],
                                          padding='VALID')
                elif layer.pooling == 'avg':
                    temp = tf.nn.avg_pool(temp,
                                          [1, pool_sup,
                                           pool_sup, 1],
                                          [1, layer.pooling_stride,
                                           layer.pooling_stride, 1],
                                          padding='VALID')
            layer_in = temp
            # Activation
            if i < n_conv_layers - 1:
                layer_in = tf.nn.relu(layer_in)
            # Dropout
            layer_in = tf.nn.dropout(layer_in, keep_prob)
            if isinstance(layer, cnn_params.GEConvolutionalLayer):
                if layer.ge_pool:
                    layer_in = GroupPool(layer.ge_type)(layer_in)
                    print('POOOLLLINNGGG!!!!!!!!!')
#                    if layer.ge_pool == 'max':
#                        layer_in = groupy_pool_group(layer_in,
#                                                     [batch_size, 1,
#                                                      1, layer.n_filters],
#                                                     h_output)
#                    elif layer.ge_pool == 'avg':
#                        raise ValueError('Average pooling not yet tested for Groupy!')

            tf.summary.histogram(layer_name + '/activations',
                                 layer_in)
            if visualize_conv:
                feat_map_summaries.append(layer_in)
            n_filters_previous_layer = layer.n_filters
            layer_ind += 1
            prev_layer = layer
            layer_in_shape = [batch_size, layer_in_size, layer_in_size,
                              layer.filter_size]
            if visualize:
                activations_to_viz.append(layer_in)
                weights_to_viz.append(weights)

    # Calculate number of nodes for fully-connected layers.
    layer_in_n_nodes = params.calculate_n_output_conv_nodes(input_size)
    conv_size = int(math.sqrt(layer_in_n_nodes / layer.n_filters))

#    # GEConvolution pooling
#    if isinstance(layer, cnn_params.GEConvolutionalLayer):
#        if layer.ge_pool:
#            if layer.ge_pool == 'max':
#                layer_in = groupy_pool_group(layer_in,
#                                             [batch_size, conv_size,
#                                              conv_size, layer.n_filters],
#                                             h_output)
#            elif layer.ge_pool == 'avg':
#                raise ValueError('Average pooling not yet tested for Groupy!')

    # Transition from convolutional to fully-connected layers via DFT
    print('layer_in_n_nodes = %d' % layer_in_n_nodes)
    prev_layer = layer
    layer = params.transition_layer
    if layer:
        layer_in = tf.nn.relu(layer_in)
        if isinstance(prev_layer, cnn_params.REConvolutionalLayer):
            conv_dim = params.calculate_conv_feature_map_dims(input_size)[-1]
            print('conv_dim = %d' % conv_dim)
            print('n_filters_previous_layer = %d' % n_filters_previous_layer)
            print('layer.n_nodes = %d' % layer.n_nodes)
            layer_name = 'conv-to-fc'
            print(layer_name + ': ' + str(layer))
            with tf.name_scope(layer_name):
                # Uncomment the following later
                weights = \
                    weight_variable_with_rotations([conv_dim,
                                                    conv_dim,
                                                    n_filters_previous_layer,
                                                    int(layer.n_nodes)],
                                                   layer.n_rotations,
                                                   layer_name + '/weights')
                biases = bias_variable([int(layer.n_nodes)],
                                       layer_name + '/biases')
                variable_summaries(biases, layer_name + '/biases')
                biases = tf.tile(biases, [layer.n_rotations])
                # Pass through weights
                temp = conv2d(layer_in, weights, 1) + biases

                # Activation
                layer_in_fft = tf.nn.relu(temp)
                # Output should be [batch_size, 1, 1, n_nodes*n_rotations], so squeeze
                layer_in_fft = tf.squeeze(layer_in_fft)

#                layer_in = tf.expand_dims(layer_in, 3)
#                layer_in = tf.expand_dims(layer_in, 4)
#                layer_in = hh_lite.conv2d_with_spaced_rotations(
#                        layer_in, conv_dim, layer.n_rotations, max_order=2)
#                layer_in_fft = tf.reshape(layer_in)
                # DFT
                layer_in, pre_activations = dft2d_transition(
                        layer_in_fft, layer.n_rotations, batch_size, conv_size,
                        n_filters_previous_layer, layer.n_nodes)
                layer_in_n_nodes = layer.n_nodes*layer.n_rotations
                if visualize:
                    activations_to_viz.append(pre_activations)
                    activations_to_viz.append(layer_in)
                    weights_to_viz.append(weights)
#                # Dropout is applied after layer
#                if layer.dropout:
#                    layer_in = tf.nn.dropout(layer_in, keep_prob)
        # If G-CNN, pool over group.
        elif isinstance(prev_layer, cnn_params.GEConvolutionalLayer):
            # Do not perform DFT transition if GE pooling was applied
            if not prev_layer.ge_pool:
                # layer_in should be [batch_size, x, x, n_filters*4], so reshape
                #   to [batch_size, x*x*n_nodes*4]
#                layer_in_fft = tf.squeeze(layer_in)
                # Reshape output to be [batch_size, x*x*n_filters, 4]
                print('GCNN + DFT')
                output = tf.transpose(layer_in, [1, 2, 3, 0])
                output = tf.reshape(output, [conv_size*conv_size*n_filters_previous_layer,
                                             layer.n_rotations, batch_size])
                output = tf.transpose(output, [2, 0, 1])
                # DFT to enforce rotational invariance
                output = tf.cast(output, tf.complex64)
                #--- retain sign of DFT value at zero frequency
                layer_fft = tf.fft2d(output)
                mask_abs = np.ones((batch_size, n_filters_previous_layer, layer.n_rotations), dtype='bool')
                mask_abs[:, 0, 0] = 0
#                mask_abs = tf.convert_to_tensor(mask_abs)
                mask_abs = tf.constant(mask_abs)
                layer_fft_abs = tf.abs(layer_fft)
                layer_fft_real = tf.real(layer_fft)
                layer_fft = tf.where(mask_abs, layer_fft_abs, layer_fft_real)
                #--- old method without is below
                #layer_in_fft = tf.abs(tf.fft2d(output))
                #layer_in_fft = layer_in_fft / (conv_size*conv_size*n_filters_previous_layer*4)
                #---
                # TODO: need to fix this normalization
                layer_in = layer_fft / (conv_size*conv_size*n_filters_previous_layer*layer.n_rotations)
                layer_in_n_nodes = conv_size*conv_size*n_filters_previous_layer *layer.n_rotations
#                # Dropout is applied after layer
#                if layer.dropout:
#                    layer_in = tf.nn.dropout(layer_in, keep_prob)
    else:
        # If fc layers, then apply relu
        if n_fc_layers > 0:
            layer_in = tf.nn.relu(layer_in)
    if isinstance(params.conv_layers[-1], cnn_params.GEConvolutionalLayer):
        if not layer and not params.conv_layers[-1].ge_pool:
            if params.conv_layers[-1].ge_type == 'C4':
                print('C4!!!!!!!!!!!!!!!')
                layer_in_n_nodes = layer_in_n_nodes*4
            elif params.conv_layers[-1].ge_type == 'D4':
                layer_in_n_nodes = layer_in_n_nodes*8

    # Reshape for fully connected layers
    layer_in = tf.reshape(layer_in, [batch_size, -1])
    # If transition layer, then apply batch norm
    if layer:
        with tf.variable_scope(layer_name + '2' + '_bn'):
            layer_in = tf.contrib.layers.batch_norm(
                    layer_in, decay=0.95, center=True, scale=True,
    #                            temp, center=True, scale=True,
                    is_training=is_training)
#    biases = bias_variable([int(layer_in_n_nodes)],
#                           layer_name + '2' + '/biases')
#    layer_in = tf.nn.relu(layer_in + biases)
    # Dropout is applied after layer
#    if layer.dropout:
#        layer_in = tf.nn.dropout(layer_in, keep_prob)
    # Fully-connected layers
    layer_ind = 0
    for i, layer in zip(range(n_fc_layers), params.fully_connected_layers):
        layer_name = 'full' + str(layer_ind)
        print(layer_name + ': ' + str(layer))
        with tf.name_scope(layer_name):
            weights = weight_variable([layer_in_n_nodes, layer.n_nodes],
                                      layer_name + '/weights')
            variable_summaries(weights, layer_name + '/weights')
            bias = bias_variable([layer.n_nodes],
                                 layer_name + '/biases')
            variable_summaries(bias, layer_name + '/biases')
            # Pass through weights
            temp = tf.matmul(layer_in, weights) + bias
            # Do not apply activation and bn to final layer
            if i < n_fc_layers - 1:
                # Batch normalization
                if layer.batch_norm:
                    with tf.variable_scope(layer_name + '_bn'):
                        temp = tf.contrib.layers.batch_norm(temp,
                                                            decay=0.95,
                                                            center=True, scale=True,
                                                            is_training=is_training)
                # Activation
                relu = tf.nn.relu(temp)
                tf.summary.histogram(layer_name + '/activations', relu)
                layer_in = relu
            else:
                layer_in = temp
            layer_ind += 1
            layer_in_n_nodes = layer.n_nodes
            if visualize:
                activations_to_viz.append(layer_in)
        # Dropout is applied after layer
#        if layer.dropout:
#            layer_in = tf.nn.dropout(layer_in, keep_prob)
    if visualize_conv:
        return layer_in, layer_in_n_nodes, feat_map_summaries
    elif visualize:
        return layer_in, layer_in_n_nodes, [activations_to_viz, weights_to_viz]
    else:
        return layer_in, layer_in_n_nodes


def loss(logits, labels, one_hot=False, loss_function='cross_entropy'):
    if loss_function == 'mse':
        logits = tf.sigmoid(logits)
        mse = tf.reduce_sum(tf.squared_difference(logits, labels))
        tf.summary.scalar('mean_squared_error', mse)
        prediction_loss = mse
    else:
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
    return train_step


def evaluation(logits, labels, one_hot):
    if one_hot:
        correct = tf.nn.in_top_k(logits, tf.argmax(labels, 1), 1)
    else:
        correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
