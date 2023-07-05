from keras.layers import Conv1D, BatchNormalization, Activation, Input, Add, UpSampling1D, GlobalAveragePooling1D, \
    Dense, Multiply, Concatenate, AveragePooling1D, TimeDistributed, GRU, Flatten
from keras.models import Model


def SeqUNet(input_shape):
    def cbr(cbr_layer_input, out_layer, kernel, stride, dilation):
        cbr_layer = TimeDistributed(Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation,
                                           strides=stride, padding="same"))(cbr_layer_input)
        cbr_layer = TimeDistributed(BatchNormalization())(cbr_layer)
        cbr_layer = TimeDistributed(Activation("relu"))(cbr_layer)
        return cbr_layer

    def se_block(se_block_layer_input, se_block_layer_n):
        se_block_layer = TimeDistributed(GlobalAveragePooling1D())(se_block_layer_input)
        se_block_layer = TimeDistributed(Dense(se_block_layer_n // 8, activation="relu"))(se_block_layer)
        se_block_layer = TimeDistributed(Dense(se_block_layer_n, activation="sigmoid"))(se_block_layer)
        x_out = Multiply()([se_block_layer_input, se_block_layer])
        return x_out

    def resblock(resblock_layer_input, resblock_layer_n, kernel, dilation, use_se=True):
        resblock_layer = cbr(resblock_layer_input, resblock_layer_n, kernel, 1, dilation)
        resblock_layer = cbr(resblock_layer, resblock_layer_n, kernel, 1, dilation)
        if use_se:
            resblock_layer = se_block(resblock_layer, resblock_layer_n)
        resblock_layer = Add()([resblock_layer_input, resblock_layer])
        return resblock_layer

    layer_n = 64
    small_kernel_size = 3
    depth = 1

    input_layer = Input(input_shape)  # 64
    input_layer_1 = TimeDistributed(AveragePooling1D(4))(input_layer)  # 16

    # Encoder
    x = cbr(input_layer, layer_n, small_kernel_size, 1, 1)  # 64
    for i in range(depth):
        x = resblock(x, layer_n, small_kernel_size, 1)
    out_0 = x

    x = cbr(x, layer_n * 2, small_kernel_size, 4, 1)  # 16
    for i in range(depth):
        x = resblock(x, layer_n * 2, small_kernel_size, 1)
    out_1 = x

    x = Concatenate()([x, input_layer_1])
    x = cbr(x, layer_n * 3, small_kernel_size, 4, 1)  # 4
    for i in range(depth):
        x = resblock(x, layer_n * 3, small_kernel_size, 1)

    # Decoder
    x = TimeDistributed(UpSampling1D(4))(x)
    x = Concatenate()([x, out_1])
    out_up_0 = cbr(x, layer_n * 3, small_kernel_size, 1, 1)

    x = TimeDistributed(UpSampling1D(4))(out_up_0)
    x = Concatenate()([x, out_0])
    out_up_1 = cbr(x, layer_n * 2, small_kernel_size, 1, 1)

    # binary classifier
    x = TimeDistributed(Conv1D(1, kernel_size=small_kernel_size, strides=1, padding="same"))(out_up_1)
    x = TimeDistributed(Flatten())(x)
    out = Activation('sigmoid')(x)

    return Model(input_layer, out)
