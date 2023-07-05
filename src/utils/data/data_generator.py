import copy
import random

import numpy as np
from keras.utils import Sequence


class SequenceDataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, input_signals, targets, sequence_batch_size, epoch_batch_size, to_fit=True, shuffle=True):
        """Initialization
        :param input_signals: all the signals
        :param epoch_batch_size: batch size at each iteration
        """
        self.input_signals: list[list[int]] = input_signals
        self.targets: list[list[int]] = targets
        self.signal_batch_size: int = sequence_batch_size
        self.epoch_batch_size: int = epoch_batch_size
        self.to_fit: bool = to_fit
        self.shuffle: bool = shuffle
        self.indexes: list[int] = [i for i in range(len(self.input_signals))]

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.input_signals) / self.epoch_batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.epoch_batch_size:(index + 1) * self.epoch_batch_size]

        # Generate data
        input_batches = self._generate_X(indexes)

        if self.to_fit:
            target = self._generate_y(indexes)
            return [input_batches], target
        else:
            return [input_batches]

    def _generate_X(self, indexes):
        # Initialization
        batches = []
        signal_batches = []

        # Generate data
        for epoch_batch_index in indexes:
            signal_batch_number = np.ceil(len(self.input_signals[epoch_batch_index]) / self.signal_batch_size)
            for signal_batch_index in range(signal_batch_number):
                # Store sample
                if (signal_batch_index + 1) * self.signal_batch_size < len(self.input_signals[epoch_batch_index]):
                    start_idx = signal_batch_index * self.signal_batch_size
                    end_idx = (signal_batch_index + 1) * self.signal_batch_size
                    temp = self.input_signals[epoch_batch_index][start_idx:end_idx]
                else:
                    temp = self.input_signals[epoch_batch_index][signal_batch_index * self.signal_batch_size:]
                    temp.extend([0 for _ in range(self.signal_batch_size - len(temp))])
                signal_batches.append(temp)
            batches.append(signal_batches)

        return batches

    def _generate_y(self, indexes):
        batches = []

        # Generate data
        for epoch_batch_index in indexes:
            target = copy.copy(self.targets[epoch_batch_index])
            target.extend([0 for _ in range(len(target) % self.signal_batch_size)])
            batches.append(target)

        return batches

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indexes)
