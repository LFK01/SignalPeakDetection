import copy
import time
import datetime
import os
import struct
import pickle
import random

import numpy as np
import tensorflow as tf
from src.utils.Signal import SignalLight
from src.definitions import ROOT_DIR

FLOAT_SIZE = 4
UNSIGNED_INT_SIZE = 1

SIGNAL_SIZE = 1024
TARGET_SIZE = 1024

DATASET_SIZE = 50000
TRAINING_PERCENTAGE = 0.7
VALIDATION_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.1

DOWNSAMPLING_STEP = 20
TRAINING_BATCH_SIZE = 100

LEARNING_RATE = 0.0005
MAX_EPOCHS = 600


def split_datasets():
    series = read_raw_files(save_pickles=False)
    splits = [TRAINING_PERCENTAGE, VALIDATION_PERCENTAGE, TEST_PERCENTAGE]

    random.shuffle(series)
    series = series[::DOWNSAMPLING_STEP]
    train_data = series[:int(splits[0] * len(series))]
    val_data = series[int(splits[0] * len(series)):int((splits[0] + splits[1]) * len(series))]
    test_data = series[int((splits[0] + splits[1]) * len(series)):]

    return train_data, val_data, test_data


def prepare_data(cut_signal_length: bool = False,
                 build_tf_datasets: bool = False):
    train_signals, val_signals, test_signals = split_datasets()

    train_input = []
    train_target = []

    mean_value: float = 0.0
    max_value: int = 0
    mean_length: int = 0

    for train_sample in train_signals:
        if cut_signal_length:
            new_data = train_sample.data[:int(train_sample.length)]
            train_input.append(new_data)
        else:
            new_data = train_sample.data
            train_input.append(new_data)

        target = [0] * 1024
        for peak in train_sample.peaks:
            target[int(round(peak))] = 1
        train_target.append(target)

        mean_value = (sum(new_data) + mean_length * mean_value) / (mean_length + len(new_data))
        mean_length += len(new_data)
        if max_value < max(new_data):
            max_value = max(new_data)

    val_input = []
    val_target = []

    for val_sample in val_signals:
        if cut_signal_length:
            new_data = val_sample.data[:int(val_sample.length)]
            val_input.append(new_data)
        else:
            new_data = val_sample.data
            val_input.append(new_data)

        target = [0] * 1024
        for peak in val_sample.peaks:
            target[int(round(peak))] = 1
        val_target.append(target)

        mean_value = (sum(new_data) + mean_length * mean_value) / (mean_length + len(new_data))
        mean_length += len(new_data)
        if max_value < max(new_data):
            max_value = max(new_data)

    test_input = []
    test_target = []

    for test_sample in test_signals:
        if cut_signal_length:
            new_data = test_sample.data[:int(test_sample.length)]
            test_input.append(new_data)
        else:
            new_data = test_sample.data
            test_input.append(new_data)
        target = [0] * 1024
        for peak in test_sample.peaks:
            target[int(round(peak))] = 1
        test_target.append(target)

        mean_value = (sum(new_data) + mean_length * mean_value) / (mean_length + len(new_data))
        mean_length += len(new_data)
        if max_value < max(new_data):
            max_value = max(new_data)

    # signal values are unsigned integers. Zero value is present in some signals so it useless to search for the minimum
    # value or consider it in the normalization
    for i in range(len(train_input)):
        train_input[i] = [(val - mean_value) / max_value for val in train_input[i]]

    for i in range(len(val_input)):
        val_input[i] = [(val - mean_value) / max_value for val in val_input[i]]

    test_input_not_normalized = copy.copy(test_input)
    for i in range(len(test_input)):
        test_input[i] = [(val - mean_value) / max_value for val in test_input[i]]

    if build_tf_datasets:
        return tf_datasets_builder(train_input, train_target, val_input, val_target, test_input,
                                   test_target, test_input_not_normalized)
    else:
        return np_datasets_builder(train_input, train_target, val_input, val_target, test_input,
                                   test_target, test_input_not_normalized)


def np_datasets_builder(train_input_list: list[list[float]],
                        train_target_list: list[list[int]],
                        val_input_list: list[list[float]],
                        val_target_list: list[list[int]],
                        test_input_list: list[list[float]],
                        test_target_list: list[list[int]],
                        test_input_not_normalized: list[list[int]]):
    np_train_input = np.asarray(train_input_list, dtype=float).reshape((-1, SIGNAL_SIZE, 1))
    np_train_target = np.asarray(train_target_list, dtype=int).reshape((-1, SIGNAL_SIZE, 1))

    np_val_input = np.asarray(val_input_list, dtype=float).reshape((-1, SIGNAL_SIZE, 1))
    np_val_target = np.asarray(val_target_list, dtype=int).reshape((-1, SIGNAL_SIZE, 1))

    np_test_input = np.asarray(test_input_list, dtype=float).reshape((-1, SIGNAL_SIZE, 1))
    np_test_target = np.asarray(test_target_list, dtype=int).reshape((-1, SIGNAL_SIZE, 1))

    return np_train_input, np_train_target, np_val_input, np_val_target, np_test_input, \
        np_test_target, test_input_not_normalized


def tf_datasets_builder(train_input_list: list[list[float]],
                        train_target_list: list[list[int]],
                        val_input_list: list[list[float]],
                        val_target_list: list[list[int]],
                        test_input_list: list[list[float]],
                        test_target_list: list[list[int]],
                        test_input_not_normalized: list[list[int]]):
    tf_train_input = tf.constant(train_input_list, dtype=tf.float32)
    tf_train_target = tf.constant(train_target_list, dtype=tf.float32)
    tf_train_input = tf.reshape(tf_train_input, [-1, SIGNAL_SIZE, 1])
    tf_train_target = tf.reshape(tf_train_target, [-1, SIGNAL_SIZE, 1])
    train_dataset = tf.data.Dataset.from_tensor_slices((tf_train_input, tf_train_target))
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(TRAINING_BATCH_SIZE)

    tf_val_input = tf.constant(val_input_list, dtype=tf.float32)
    tf_val_target = tf.constant(val_target_list, dtype=tf.float32)
    tf_val_input = tf.reshape(tf_val_input, [-1, SIGNAL_SIZE, 1])
    tf_val_target = tf.reshape(tf_val_target, [-1, SIGNAL_SIZE, 1])
    val_dataset = tf.data.Dataset.from_tensor_slices((tf_val_input, tf_val_target))
    val_dataset = val_dataset.batch(1)

    tf_test_input = tf.constant(test_input_list, dtype=tf.float32)
    tf_test_target = tf.constant(test_target_list, dtype=tf.float32)
    tf_test_input = tf.reshape(tf_test_input, [-1, SIGNAL_SIZE, 1])
    tf_test_target = tf.reshape(tf_test_target, [-1, SIGNAL_SIZE, 1])
    test_dataset = tf.data.Dataset.from_tensor_slices((tf_test_input, tf_test_target))
    test_dataset = test_dataset.batch(1)

    return train_dataset, val_dataset, test_dataset, test_input_not_normalized


def load_pickles() -> list[SignalLight]:
    filepath = os.path.join(ROOT_DIR, 'storage', 'pickled_objects', 'pickle_saved_signal_parsed_2023_06_26_17_13')
    with open(filepath, 'rb') as input_file:
        saved_signals: list[SignalLight] = pickle.load(input_file)
    return saved_signals


def read_raw_files(save_pickles: bool = False):
    start_time = time.time()
    filename = os.path.join(ROOT_DIR, 'storage', 'task_data_v2_50k', 'info.raw')
    info_values: list[float] = []
    with open(filename, 'rb') as f:
        while True:
            byte_read = f.read(FLOAT_SIZE)
            if not byte_read:
                break
            float_read = struct.unpack('f', byte_read)[0]
            info_values.append(float_read)

    print('Reading info.raw time: {} s'.format(time.time() - start_time))

    start_time = time.time()

    filename = os.path.join(ROOT_DIR, 'storage', 'task_data_v2_50k', 'signal.raw')
    signal_values: list[int] = []
    with open(filename, 'rb') as f:
        while True:
            byte_read = f.read(UNSIGNED_INT_SIZE)
            if not byte_read:
                break
            float_read = struct.unpack('B', byte_read)[0]
            signal_values.append(float_read)

    print('Reading signal.raw time: {} s'.format(time.time() - start_time))

    start_time = time.time()

    signals: list[SignalLight] = []
    signal_index = 0
    while signal_index * 11 < len(info_values):
        info_values_index = signal_index * 11

        sig_len = info_values[info_values_index]
        peak_num = info_values[info_values_index + 1]
        peaks: list[float] = []
        for peak in range(int(peak_num)):
            peak_pos = info_values[info_values_index + 2 + peak * 3]
            peaks.append(peak_pos)

        data = signal_values[signal_index * 1024:(signal_index + 1) * 1024]

        signals.append(SignalLight(int(sig_len), peaks, data))
        signal_index += 1

    declared_length = 0
    saved_length = 0

    for sig in signals:
        declared_length += sig.length
        saved_length += len(sig.data)

    print('Elapsed Time: {} s'.format(time.time() - start_time))

    if save_pickles:
        now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')

        new_pickle_file = os.path.join(ROOT_DIR, 'storage', 'pickled_objects',
                                       'pickle_saved_signal_parsed_{}'.format(now))
        with open(new_pickle_file, 'wb') as out:
            pickle.dump(signals, out)

    return signals


if __name__ == '__main__':
    pass
