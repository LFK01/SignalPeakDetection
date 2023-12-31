import os
import datetime
import random

import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import Model
from sklearn.metrics import precision_score, recall_score
from src.model.UNet import UNetLight, UNetAdvanced
from src.model.SequenceUNet import SeqUNet
from src.model.loss import DiceLoss
from src.utils.data.data_utils import prepare_data, DATASET_SIZE, SIGNAL_SIZE, LEARNING_RATE, TRAINING_BATCH_SIZE, \
    DOWNSAMPLING_STEP, TRAINING_PERCENTAGE, MAX_EPOCHS
from src.utils.data.data_generator import SequenceDataGenerator
from src.definitions import ROOT_DIR

TENSORBOARD_LOG_DIR = os.path.join(ROOT_DIR, 'storage', 'tensorboard_logs')
CHECKPOINT_ROOT = os.path.join(ROOT_DIR, 'storage', 'checkpoints')
RANDOM_SEED = 1234


def train():
    train_input, train_target, val_input, val_target, \
        test_input, test_target, test_input_not_normalized = prepare_data(cut_signal_length=False,
                                                                          build_tf_datasets=False)

    train_model = UNetLight(input_shape=(SIGNAL_SIZE, 1))
    print(train_model.summary())

    tensorboard_callback = TensorBoard(log_dir=TENSORBOARD_LOG_DIR, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15)

    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    check_dir = os.path.join(CHECKPOINT_ROOT, 'check_{}'.format(now))
    if not os.path.exists(check_dir):
        os.mkdir(check_dir)

    checkpoints = ModelCheckpoint(filepath=os.path.join(check_dir, 'check'),
                                  monitor='loss',
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='min')

    train_model.compile(loss=DiceLoss(),
                        optimizer=Adam(learning_rate=LEARNING_RATE))

    train_model.fit(x=train_input,
                    y=train_target,
                    epochs=MAX_EPOCHS,
                    batch_size=TRAINING_BATCH_SIZE,
                    validation_data=(val_input, val_target),
                    callbacks=[tensorboard_callback, early_stopping, checkpoints])

    train_model.load_weights(os.path.join(check_dir, 'check'))

    saved_model_path = save_model(train_model)

    test(saved_model_path)

    return train_model


def save_model(new_model: Model):
    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    saved_model_path = os.path.join(ROOT_DIR,  'storage/saved_models', 'model_{}'.format(now))
    new_model.save(saved_model_path)

    return saved_model_path


def test(filepath):
    test_model = keras.models.load_model(filepath, custom_objects={'DiceLoss': DiceLoss}, compile=False)
    _, _, _, _, test_input, test_target, test_input_not_normalized = prepare_data()
    predictions = test_model.predict(test_input)

    print('Dice Score: {}'.format(DiceLoss.compute_test(test_target, predictions)))
    print('Precision: {}'.format(precision_score(test_target.flatten(),
                                                 predictions.flatten().round(0).astype(int))
                                 ))
    print('Recall: {}'.format(recall_score(test_target.flatten(),
                                           predictions.flatten().round(0).astype(int))
                              ))
    nrows = 3
    ncols = 3

    picks = []
    i = 0
    while len(picks) < ncols*nrows:
        peaks = [i for i, x in enumerate(test_target[i]) if x == 1]
        if len(peaks) > 2:
            picks.append(i)
        i += 1

    signal_lines = []
    pred_line = None
    target_line = None

    plt.figure(1)
    fig1, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))

    for nrow in range(nrows):
        for ncol in range(ncols):
            axs[nrow, ncol].set_title('Signal {}'.format(picks[nrow*3+ncol]))
            signal_line, = axs[nrow, ncol].plot(test_input_not_normalized[picks[nrow*3+ncol]],
                                                label='signal', c='black', alpha=0.45)
            signal_lines.append(signal_line)

            peaks = [i for i, x in enumerate(test_target[picks[nrow*3+ncol]]) if x == 1]
            for peak in peaks:
                target_line = axs[nrow, ncol].axvline(peak, c='cornflowerblue', label='target')

            preds = [i for i, x in enumerate(predictions[picks[nrow*3+ncol]]) if x >= 0.5]
            for pred in preds:
                pred_line = axs[nrow, ncol].axvline(pred, c='crimson', linestyle='dotted', label='prediction')

            axs[nrow, ncol].set_xlim(min(peaks)-20, max(peaks)+20)

    fig1.suptitle('Model peak predictions on 3 peaks signals')
    axs[2, 1].legend(handles=[signal_lines[0], target_line, pred_line], labels=['signal', 'target', 'prediction'],
                     loc='upper center', bbox_to_anchor=(0.5, -0.22), ncols=3)
    fig1.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, 'images', '9_graphs'))
    plt.close()

    pred_line = None
    target_line = None

    plt.figure(2, figsize=(15, 10))
    plt.title('Signal {}'.format(picks[3]))
    signal_line, = plt.plot(test_input_not_normalized[picks[3]], label='signal', c='black', alpha=0.45)
    peaks = [i for i, x in enumerate(test_target[picks[3]]) if x == 1]
    for peak in peaks:
        target_line = plt.axvline(peak, c='cornflowerblue', label='target')
    plt.xlim(min(peaks) - 20, max(peaks) + 20)
    preds = [i for i, x in enumerate(predictions[picks[3]]) if x >= 0.5]
    for pred in preds:
        pred_line = plt.axvline(pred, c='crimson', linestyle='dotted', label='prediction')
    plt.legend(handles=[signal_line, target_line, pred_line], labels=['signal', 'target', 'prediction'],
               loc='upper center', bbox_to_anchor=(0.5, -0.05), ncols=3)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, 'images', '9_graphs', 'signal_detail'))
    plt.show()


if __name__ == '__main__':
    random.seed(1234)
    tf.random.set_seed(1234)

    train()
