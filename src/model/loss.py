import tensorflow as tf
import keras.backend as K
from keras.losses import Loss


class DiceLoss(Loss):
    def __init__(self, smooth=0, name='dice_loss'):
        super().__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersections = K.sum(y_true_f * y_pred_f)
        dice = (2 * intersections) / (K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)))
        return 1 - dice

    @staticmethod
    def compute_test(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersections = K.sum(y_true_f * y_pred_f)
        dice = (2 * intersections) / (K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)))
        return dice
