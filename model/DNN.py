import numpy as np
from keras.layers import Dense, Conv1D, BatchNormalization, Activation, AveragePooling1D, \
    GlobalAveragePooling1D, Lambda, Input, Concatenate, Add, UpSampling1D, Multiply
from keras.models import Model


def DNN(input_shape=(None, 1)):
    layer_n = 16
    kernel_size = 7
    dilation = 1
    stride = 1

    input_layer = Input(input_shape)  # 1024

    # ENCODER
    x = Conv1D(layer_n, kernel_size=kernel_size, dilation_rate=dilation, strides=stride, padding="same")(input_layer)
    x = Conv1D(layer_n, kernel_size=kernel_size, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=4)(x)  # 256

    x = Conv1D(layer_n * 2, kernel_size=kernel_size, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = Conv1D(layer_n * 2, kernel_size=kernel_size, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=4)(x)  # 64

    x = Conv1D(layer_n * 4, kernel_size=kernel_size, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = Conv1D(layer_n * 4, kernel_size=kernel_size, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=4)(x)  # 16

    x = Conv1D(layer_n * 8, kernel_size=3, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = Conv1D(layer_n * 8, kernel_size=3, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=4)(x)  # 4

    x = Conv1D(layer_n * 16, kernel_size=3, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = Conv1D(layer_n * 16, kernel_size=3, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=4)(x)  # 1

    out = Dense(64)(x)
    out = Dense(3)(out)

    model = Model(input_layer, out)

    return model
