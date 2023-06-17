import numpy as np
from keras.utils import Sequence
from keras.utils import pad_sequences


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, data, targets, to_fit=True, batch_size=32):
        """Initialization
        :param data: full signal
        :param to_fit: peaks to be identified
        :param batch_size: batch size at each iteration
        """
        self.data = data
        self.targets = targets
        self.to_fit = to_fit
        self.batch_size = batch_size

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """

        # Generate data
        X = self._generate_X(self.__len__())

        if self.to_fit:
            y = self._generate_y(self.__len__())
            return [X], y
        else:
            return [X]

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_X(self, batch_number):
        """Generates data containing batch_size images
        :param batch_number: number of batches to be produced
        :return: batch of images
        """
        # Initialization
        batches = []

        # Generate data
        for i in range(batch_number):
            # Store sample
            if (i+1)*self.batch_size < len(self.data):
                temp = self.data[i*self.batch_size:(i+1)*self.batch_size]
            else:
                temp = self.data[i*self.batch_size:]
            batches.append(temp)

        result = pad_sequences(batches, value=0, padding='post')

        return result

    def _generate_y(self, batch_number):
        y = []

        # Generate data
        for i in range(batch_number):
            # Store sample
            if (i+1)*self.batch_size < len(self.targets):
                y.append(self.targets[i*self.batch_size:(i+1)*self.batch_size])
            else:
                y.append(self.targets[i*self.batch_size:])

        y = pad_sequences(y, value=0, padding='post')

        return y
