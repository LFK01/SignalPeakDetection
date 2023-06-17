import time
import datetime
import os
import struct
import pickle
import random
import numpy as np
from utils.Signal import SignalLight
from definitions import ROOT_DIR

FLOAT_SIZE = 4
UNSIGNED_INT_SIZE = 1
SIGNAL_SIZE = 1024
TARGET_SIZE = 1024


def split_datasets():
    series = load_pickles()
    splits = [0.7, 0.2, 0.1]

    random.shuffle(series)
    train_data = series[:int(splits[0] * len(series))]
    val_data = series[int(splits[0] * len(series)):int((splits[0] + splits[1]) * len(series))]
    test_data = series[int((splits[0] + splits[1]) * len(series)):]

    return train_data, val_data, test_data


def prepare_data():
    train, val, test = split_datasets()

    train_input = []
    train_target = []
    for train_sample in train:
        train_input.append(train_sample.data)
        target = [0]*1024
        for peak in train_sample.peaks:
            target[int(peak)] = 1
        train_target.append(target)

    val_input = []
    val_target = []

    for val_sample in val:
        val_input.append(val_sample.data)
        target = [0] * 1024
        for peak in val_sample.peaks:
            target[int(peak)] = 1
        val_target.append(target)

    test_input = []
    test_target = []

    for test_sample in test:
        test_input.append(test_sample.data)
        target = [0] * 1024
        for peak in test_sample.peaks:
            target[int(peak)] = 1
        test_target.append(target)

    np_train_input = np.asarray(train_input, dtype=int)
    np_train_input = (np_train_input - np_train_input.mean()) / (np.max(np_train_input) - np.min(np_train_input))
    np_train_input = np.reshape(np_train_input, (-1, SIGNAL_SIZE, 1))

    np_train_target = np.asarray(train_target, dtype=float)
    np_train_target = np.reshape(np_train_target, (-1, TARGET_SIZE, 1))

    np_val_input = np.asarray(val_input, dtype=int)
    np_val_input = (np_val_input - np_val_input.mean()) / (np.max(np_val_input) - np.min(np_val_input))
    np_val_input = np.reshape(np_val_input, (-1, SIGNAL_SIZE, 1))

    np_val_target = np.asarray(val_target, dtype=float)
    np_val_target = np.reshape(np_val_target, (-1, TARGET_SIZE, 1))

    np_test_input = np.asarray(test_input, dtype=int)
    np_test_input_not_normalized = np_test_input
    np_test_input = (np_test_input - np_test_input.mean()) / (np.max(np_test_input) - np.min(np_test_input))
    np_test_input = np.reshape(np_test_input, (-1, SIGNAL_SIZE, 1))

    np_test_target = np.asarray(test_target, dtype=float)
    np_test_target = np.reshape(np_test_target, (-1, TARGET_SIZE, 1))

    return np_train_input, np_train_target, np_val_input, np_val_target, np_test_input, np_test_target,\
        np_test_input_not_normalized


def load_pickles() -> list[SignalLight]:
    with open(os.path.join(ROOT_DIR, 'saved_objects', 'pickle_saved_signal_parsed_2023_06_13_20_54'), 'rb') as input_file:
        saved_signals = pickle.load(input_file)
    return saved_signals


def save_data():
    start_time = time.time()
    filename = os.path.join(ROOT_DIR, 'task_data_v2_50k', 'info.raw')
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

    filename = os.path.join(ROOT_DIR, 'task_data_v2_50k', 'signal.raw')
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
    i = 0
    while len(info_values) > 0:
        if i % 100 == 0:
            print('i: {}, signal_len: {}, time: {} s'.format(i // 100, len(signal_values), time.time() - start_time))
        sig_len = info_values[0]
        peak_num = info_values[1]
        peaks: list[float] = []
        for peak in range(int(peak_num)):
            peak_pos = info_values[2 + peak * 3]
            peaks.append(peak_pos)

        data = signal_values[:1024]

        signals.append(SignalLight(int(sig_len), peaks, data))

        signal_values = signal_values[1024:]

        info_values = info_values[2 + 3 * 3:]
        i += 1

    declared_length = 0
    saved_length = 0

    for sig in signals:
        declared_length += sig.length
        saved_length += len(sig.data)

    print('declared: {} saved: {}'.format(declared_length, saved_length))
    print('Elapsed Time: {}'.format(time.time() - start_time))

    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')

    with open(os.path.join(ROOT_DIR, 'saved_objects', 'pickle_saved_info_raw_{}'.format(now)), 'wb') as out:
        pickle.dump(info_values, out, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(ROOT_DIR, 'saved_objects', 'pickle_saved_signal_raw_{}'.format(now)), 'wb') as out:
        pickle.dump(signal_values, out, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(ROOT_DIR, 'saved_objects', 'pickle_saved_signal_parsed_{}'.format(now)), 'wb') as out:
        pickle.dump(signals, out, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    train_in, train_g, val_in, val_g, test_in, test_g = prepare_data()
    print('train shape: {}'.format(train_in.shape))
    print('End')
