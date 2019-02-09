from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import cPickle as pkl
import random

from cfnet import network
from cfnet import data
import cnn_params

import os
import time

from scipy import ndimage


random.seed(40)


def softmax(x):
    '''Compute softmax values for each sets of scores in x.'''
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def placeholder_inputs(batch_size, img_size, img_depth, n_classes):
    '''Create placeholders for images and labels.'''
    images_placeholder = tf.placeholder(tf.float32,
                                        shape=(batch_size, img_size,
                                               img_size, img_depth))
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, n_classes))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl, is_training_pl, keep_prob_pl,
                   batch_size, is_training, keep_prob, zero_pad=0, repeats=True):
    '''Fill the feed_dict.

        zero_pad : int
            Last zero_pad elements of batch_size will be zero.
        repeats : bool
            Fetch batch from repeated epoch or not. Repeated epoch is used for
            training loss, no repeats is used for evaluation.
    '''
    if zero_pad > 0:
        images_feed = np.zeros((batch_size + zero_pad, data_set._image_size,
                                data_set._image_size, data_set._image_depth),
                               dtype='float32')
        labels_feed = np.zeros((batch_size + zero_pad, data_set.n_classes), dtype='uint8')
        if repeats:
            images_feed_temp, labels_feed_temp = data_set.next_batch(batch_size)
        else:
            images_feed_temp, labels_feed_temp = data_set.next_batch_no_repeats(batch_size)
        images_feed[0: batch_size] = images_feed_temp
        labels_feed[0: batch_size] = labels_feed_temp
    else:
        if repeats:
            images_feed, labels_feed = data_set.next_batch(batch_size)
        else:
            images_feed, labels_feed = data_set.next_batch_no_repeats(batch_size)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
        is_training_pl: is_training,
        keep_prob_pl: keep_prob
    }
    return feed_dict


def do_eval(sess,
            logits,
            images_placeholder,
            labels_placeholder,
            is_training_placeholder,
            keep_prob_placeholder,
            data_set,
            batch_size,
            repeats=True,
            classes_to_merge=None,
            classes_to_ignore=None,
            threshold_offset=0.0,
            show_cm_plot=False,
            metric='accuracy'):
    '''Runs one evaluation against the full epoch of data.'''
    if repeats:
        data_set.reset_epoch()
        num_examples = data_set._n_examples_in_epoch
    else:
        data_set.reset_no_repeats()
        num_examples = int(data_set._n_examples)
    steps_per_epoch = num_examples // batch_size
    remainder_step = num_examples % batch_size

    # Output will be of shape [# examples, # classes]
    y_dtype = 'float32'
    #y_hat = np.zeros((num_examples,), dtype=y_dtype)
    #y = np.zeros((num_examples,), dtype=y_dtype)
    y_hat = np.zeros((num_examples, data_set._n_classes), dtype=y_dtype)
    y = np.zeros((num_examples, data_set._n_classes), dtype=y_dtype)

    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder,
                                   is_training_placeholder,
                                   keep_prob_placeholder,
                                   batch_size,
                                   is_training=False,
                                   keep_prob=1.0,
                                   repeats=repeats)
        logits_eval = sess.run(logits, feed_dict=feed_dict)
        if data_set._n_classes == 2:
            y_hat[step*batch_size: (step+1)*batch_size] = \
                (logits_eval[:, 0] < (logits_eval[:, 1] + threshold_offset)).astype('uint8')
        else:
            #y_hat[step*batch_size: (step+1)*batch_size] = \
            #    np.argmax(logits_eval, axis=1)
            y_hat[step*batch_size: (step+1)*batch_size] = \
                logits_eval
        #y[step*batch_size: (step+1)*batch_size] = np.argmax(feed_dict[labels_placeholder], axis=1)
        y[step*batch_size: (step+1)*batch_size] = feed_dict[labels_placeholder]
    if remainder_step > 0:
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder,
                                   is_training_placeholder,
                                   keep_prob_placeholder,
                                   batch_size=remainder_step,
                                   is_training=False,
                                   keep_prob=1.0,
                                   zero_pad=batch_size-remainder_step,
                                   repeats=repeats)
        logits_eval = sess.run(logits, feed_dict=feed_dict)
        if data_set._n_classes == 2:
            y_hat[(step+1)*batch_size:] = \
                (logits_eval[:, 0] < (logits_eval[:, 1] + threshold_offset)).astype('uint8')[0:remainder_step]
        else:
            #y_hat[(step+1)*batch_size:] = \
            #    np.argmax(logits_eval, axis=1)[0:remainder_step]
            y_hat[(step+1)*batch_size:] = \
                logits_eval[0:remainder_step]
        #y[(step+1)*batch_size:] = np.argmax(feed_dict[labels_placeholder][0:remainder_step], axis=1)
        y[(step+1)*batch_size:] = feed_dict[labels_placeholder][0:remainder_step]

    y_hat = sigmoid(y_hat)
    return prediction_evaluation(y_hat, y, data_set._n_classes,
                                 range(data_set._n_classes),
                                 data_set._label_list,
                                 classes_to_merge, classes_to_ignore,
                                 show_cm_plot, metric)
    #return prediction_evaluation(np.array(y_hat), np.array(y), data_set._n_classes,
    #                             range(data_set._n_classes))


def prediction_evaluation(y_hat, y, n_classes=2, label_list=[0, 1],
                          label_list_names=None, classes_to_merge=None,
                          classes_to_ignore=None, show_plot=False,
                          metric='accuracy'):

    if classes_to_merge or classes_to_ignore:
        label_list_names_temp = []
        n_classes_new = y_hat.shape[1]
        if classes_to_merge:
            unique_classes_to_merge = [item for sublist in classes_to_merge for item in sublist]
            for ctm in classes_to_merge:
                n_classes_new -= len(ctm) - 1
        if classes_to_ignore:
            n_classes_new -= len(classes_to_ignore)
        y_hat_temp = np.zeros((y_hat.shape[0], n_classes_new))
        y_temp = np.zeros((y.shape[0], n_classes_new), dtype='uint8')
        i_new = 0
        for i in range(n_classes):
            if (not classes_to_merge or (classes_to_merge and i not in unique_classes_to_merge)) \
                    and (not classes_to_ignore or (classes_to_ignore and i not in classes_to_ignore)):
                print(i_new, i)
                y_hat_temp[:, i_new] = y_hat[:, i]
                y_temp[:, i_new] = y[:, i]
                label_list_names_temp.append(label_list_names[i])
                i_new += 1
        if classes_to_merge:
            for ctm in classes_to_merge:
                print(i_new, ctm)
                new_label_list = []
                for c in ctm:
                    new_label_list.append(label_list_names[c])
                    y_hat_temp[:, i_new] += y_hat[:, c]
                    y_temp[:, i_new] = np.logical_or(y_temp[:, i_new], y[:, c])
                label_list_names_temp.append(' / '.join(new_label_list))
                i_new += 1
        if classes_to_ignore:
            y_ind = np.ones((y.shape[0],), dtype='bool')
            for cti in classes_to_ignore:
                y_ind = np.logical_and(y_ind, y[:, cti] == 0)
        y_hat = y_hat_temp[y_ind]
        y = y_temp[y_ind]
        label_list = range(n_classes_new)
        label_list_names = label_list_names_temp
    avg_precision = average_precision_score(y, y_hat)
    avg_precision_weighted = average_precision_score(y, y_hat, average='weighted')
    y_not_onehot = np.argmax(y, axis=1)
    y_hat_not_onehot = np.argmax(y_hat, axis=1)
    C = confusion_matrix(y_not_onehot, y_hat_not_onehot, labels=label_list)
    if show_plot:
        df_cm = pd.DataFrame(C, label_list_names, label_list_names)
        plt.figure(figsize=(11, 8))
        sn.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
        plt.show()
    accuracy = np.sum(C.diagonal()) / float(np.sum(C))
    accuracy_weighted = np.average(C.diagonal() / np.sum(C, axis=1).astype('float'))
    print(C)
    print('Accuracy = %.6f\n' % accuracy)
    print('Weighted Accuracy = %.6f\n' % accuracy_weighted)
    print('Average Precision = %.6f\n' % avg_precision)
    print('Average Precision Weighted = %.6f\n' % avg_precision_weighted)
    return({'conf_matrix': C, 'accuracy': accuracy,
            'accuracy_weighted': accuracy_weighted,
            'avg_precision': avg_precision,
            'avg_precision_weighted': avg_precision_weighted})


def evaluate_on_dataset(model_filename, data_set,
                        param_filename, param_ind, repeats=False,
                        classes_to_merge=None, classes_to_ignore=None,
                        show_cm_plot=True, metric='accuracy',
                        radial_split=False):
    param_list = cnn_params.load_params(param_filename)
    params = param_list[param_ind]
    input_size = data_set._image_size
    input_depth = data_set._image_depth

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = \
            placeholder_inputs(params.batch_size, input_size, input_depth, data_set._n_classes)
        keep_prob_placeholder = tf.placeholder(tf.float32)
        is_training_placeholder = tf.placeholder(tf.bool)
        with tf.variable_scope("RiCNN") as scope:
            pre_logits, n_nodes = network.inference(
                    images_placeholder, params, input_size, input_depth,
                    params.batch_size, data_set._n_classes,
                    is_training_placeholder, keep_prob_placeholder)
        logits = network.output_layer(pre_logits, params,
                                      data_set._n_classes,
                                      n_nodes)
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, model_filename)
        print('DataSet Eval:')
        return do_eval(sess,
                       logits,
                       images_placeholder,
                       labels_placeholder,
                       is_training_placeholder,
                       keep_prob_placeholder,
                       data_set,
                       params.batch_size,
                       repeats=repeats,
                       classes_to_merge=classes_to_merge,
                       classes_to_ignore=classes_to_ignore,
                       show_cm_plot=show_cm_plot,
                       metric=metric)


def train(param_filename, model_dir, summary_dir, data_train, data_valid,
          results_dir='',
          n_iterations=1, metric='accuracy',
          filename_suffix='',
          save_models=True, max_to_keep=0, save_summaries=True,
          dft_flip=False, radial_split=False,
          loss_function='cross_entropy',
          evaluate_interval=None,
          train_interval=None):
    '''Train the network.

        param_filename - string:
            Filename of the parameter file describing the network.
        model_dir - string:
            Where to save the model files.
        summary_dir - string:
            Where to save the variable summaries.
        n_iterations - int:
            How many times to retrain the model (usually simply 1).
        data_set - list of DataSet's [data_train, data_valid]:
            Train and validation data sets of .
        metric - {'accuracy', 'rmse'}:
            Metric for evaluating network.
    '''
    # Load parameter list from parameters file.
    param_list = cnn_params.load_params(param_filename)
    filename_prefix = os.path.splitext(os.path.split(param_filename)[1])[0]
    input_size = data_train._image_size
    input_depth = data_train._image_depth

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(summary_dir):
        os.makedirs(summary_dir)
    if results_dir != '':
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

    # Iterate over sets of parameters in parameter file.
    for param_ind, params in zip(range(len(param_list)), param_list):
        # Determine interval at which to evaluate model
        if not evaluate_interval:
            evaluate_interval =\
                    data_train._n_examples_in_epoch / params.batch_size
            print("# batches per epoch = " + str(evaluate_interval))
        if not train_interval:
            train_interval =\
                    data_train._n_examples_in_epoch / params.batch_size
        evaluate_interval = int(evaluate_interval)
        train_interval = int(train_interval)
        with tf.Graph().as_default():
            # Placeholder initialization
            global_step = tf.Variable(0, trainable=False)
            images_placeholder, labels_placeholder = \
                placeholder_inputs(params.batch_size, input_size, input_depth,
                                   data_train._n_classes)
            keep_prob_placeholder = tf.placeholder(tf.float32)
            is_training_placeholder = tf.placeholder(tf.bool)
            # Model configuration
            with tf.variable_scope("RiCNN") as scope:
                pre_logits, n_nodes = network.inference(
                        images_placeholder, params, input_size, input_depth,
                        params.batch_size, data_train._n_classes,
                        is_training_placeholder, keep_prob_placeholder)
            logits = pre_logits
#            logits = network.output_layer(pre_logits, params,
#                                          data_train._n_classes,
#                                          n_nodes)
            loss_ = network.loss(logits, labels_placeholder,
                                 data_train._one_hot, loss_function)
            # Add regularization to loss
            loss_ = loss_ + tf.add_n(
                    tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            # Training updates
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = network.training(loss_, global_step,
                                            params.learning_rate,
                                            train_interval)
            # Model saver and summary
            saver = tf.train.Saver(max_to_keep=max_to_keep)
            summary_op = tf.summary.merge_all()

            total_params = 0
            # Check number of trainable params
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_params += variable_parameters
            print("Number of trainable parameters: %d" % (total_params))

            # Train the same model several times
            print('Starting session...')
            sess = tf.Session()
            for iteration in range(n_iterations):
                model_name = filename_prefix + filename_suffix + '_' +\
                        str(param_ind) + '_' + str(iteration)
                # Pickle files for results
                f_train = open(os.path.join(results_dir,
                                            model_name + '_train_results.pkl'),
                               'wb')
                f_valid = open(os.path.join(results_dir,
                                            model_name + '_valid_results.pkl'),
                               'wb')
                train_results = []
                valid_results = []

                print('Initializing variables...')
                sess.run(tf.initialize_all_variables())
                if save_summaries:
                    print('Initializing summary writer...')
                    summary_writer = tf.summary.FileWriter(
                            os.path.join(summary_dir, model_name), sess.graph)

                if save_models:
                    saver.save(sess, os.path.join(model_dir, model_name),
                               global_step=0)

                # Iterate over batches until max_steps is reached
                for step in xrange(params.max_steps):
                    start_time = time.time()
                    feed_dict = fill_feed_dict(data_train,
                                               images_placeholder,
                                               labels_placeholder,
                                               is_training_placeholder,
                                               keep_prob_placeholder,
                                               params.batch_size,
                                               is_training=True,
                                               keep_prob=params.dropout_prob)
                    _, loss_value = sess.run([train_op, loss_],
                                             feed_dict=feed_dict)
                    duration = time.time() - start_time
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    # Save variable summaries and print loss
                    if (step + 1) % 10 == 0:
                        num_examples_per_step = params.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)
                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print(format_str % (datetime.now(), step, loss_value,
                              examples_per_sec, sec_per_batch))

                        if save_summaries:
                            summary_str = sess.run(summary_op, feed_dict=feed_dict)
                            summary_writer.add_summary(summary_str, step)

                    # Save a checkpoint and evaluate the model periodically.
                    if (step + 1) % evaluate_interval == 0 or (step + 1) == params.max_steps:
                        if save_models:
                            saver.save(sess, os.path.join(model_dir, model_name),
                                       global_step=step)
#                        # Evaluate against the training set.
#                        print("Step: " + str(step))
#                        print('Training Data Eval:')
#                        results = do_eval(sess,
#                                          logits,
#                                          images_placeholder,
#                                          labels_placeholder,
#                                          is_training_placeholder,
#                                          keep_prob_placeholder,
#                                          data_train,
#                                          params.batch_size,
#                                          repeats=False,
#                                          metric=metric)
#                        train_results.append(results)
                    if (step + 1) % evaluate_interval == 0 or (step + 1) == params.max_steps:
                        # Evaluate against the validation set.
                        print("Step: " + str(step))
                        print('Validation Data Eval:')
                        results = do_eval(sess,
                                          logits,
                                          images_placeholder,
                                          labels_placeholder,
                                          is_training_placeholder,
                                          keep_prob_placeholder,
                                          data_valid,
                                          params.batch_size,
                                          repeats=False,
                                          metric=metric)
                        valid_results.append(results)

                # Save the results.
                pkl.dump(train_results, f_train)
                pkl.dump(valid_results, f_valid)
                f_train.close()
                f_valid.close()
            sess.close()
    return
