from keras.layers import Conv1D, BatchNormalization, Activation, Input, Add, UpSampling1D, MaxPooling1D, \
    Conv1DTranspose, GlobalAveragePooling1D, Dense, Multiply, Concatenate, AveragePooling1D
from keras.models import Model


def UNet(input_shape=(None, 1)):
    kernel_size = 3

    input_layer = Input(shape=input_shape)  # 1024
    # Entry block
    x = Conv1D(32, kernel_size=kernel_size, strides=4, padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    previous_block_activation = x  # Set aside residual

    # ENCODER
    for filters in [32, 128, 256]:
        x = Activation('relu')(x)
        x = Conv1D(filters, kernel_size=kernel_size, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = Conv1D(filters, kernel_size=kernel_size, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling1D(pool_size=4)(x)

        residual = Conv1D(filters, 1, strides=4, padding="same")(previous_block_activation)

        x = Add()([x, residual])
        previous_block_activation = x

    # DECODER
    for filters in [256, 128, 64, 32]:
        x = Activation("relu")(x)
        x = Conv1DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv1DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = UpSampling1D(size=4)(x)

        # Project residual
        residual = UpSampling1D(4)(previous_block_activation)
        residual = Conv1D(filters, 1, padding="same")(residual)
        x = Add()([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv1D(filters=1, kernel_size=3, activation="softmax", padding="same")(x)

    model = Model(input_layer, outputs)
    return model


def UNetAdvanced(input_shape=(None, 1)):
    def cbr(cbr_layer_input, out_layer, kernel, stride, dilation):
        cbr_layer = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(cbr_layer_input)
        cbr_layer = BatchNormalization()(cbr_layer)
        cbr_layer = Activation("relu")(cbr_layer)
        return cbr_layer

    def se_block(se_block_layer_input, layer_n):
        se_block_layer = GlobalAveragePooling1D()(se_block_layer_input)
        se_block_layer = Dense(layer_n // 8, activation="relu")(se_block_layer)
        se_block_layer = Dense(layer_n, activation="sigmoid")(se_block_layer)
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
    large_kernel_size = 7
    small_kernel_size = 3
    depth = 2

    input_layer = Input(input_shape)
    input_layer_1 = AveragePooling1D(4)(input_layer)
    input_layer_2 = AveragePooling1D(16)(input_layer)

    # Encoder
    x = cbr(input_layer, layer_n, large_kernel_size, 1, 1)  # 1024
    for i in range(depth):
        x = resblock(x, layer_n, large_kernel_size, 1)
    out_0 = x

    x = cbr(x, layer_n * 2, large_kernel_size, 4, 1)  # 256
    for i in range(depth):
        x = resblock(x, layer_n * 2, large_kernel_size, 1)
    out_1 = x

    x = Concatenate()([x, input_layer_1])
    x = cbr(x, layer_n * 3, small_kernel_size, 4, 1)
    for i in range(depth):
        x = resblock(x, layer_n * 3, small_kernel_size, 1)
    out_2 = x

    x = Concatenate()([x, input_layer_2])
    x = cbr(x, layer_n * 4, small_kernel_size, 4, 1)
    for i in range(depth):
        x = resblock(x, layer_n * 4, small_kernel_size, 1)

    # Decoder
    x = UpSampling1D(4)(x)
    x = Concatenate()([x, out_2])
    x = cbr(x, layer_n * 3, small_kernel_size, 1, 1)

    x = UpSampling1D(4)(x)
    x = Concatenate()([x, out_1])
    x = cbr(x, layer_n * 2, small_kernel_size, 1, 1)

    x = UpSampling1D(4)(x)
    x = Concatenate()([x, out_0])
    x = cbr(x, layer_n, large_kernel_size, 1, 1)

    # binary classifier
    x = Conv1D(1, kernel_size=large_kernel_size, strides=1, padding="same")(x)
    out = Activation('sigmoid')(x)

    return Model(input_layer, out)


def UNetLight(input_shape=(None, 1)):
    def cbr(cbr_layer_input, out_layer, kernel, stride, dilation):
        cbr_layer = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(cbr_layer_input)
        cbr_layer = BatchNormalization()(cbr_layer)
        cbr_layer = Activation("relu")(cbr_layer)
        return cbr_layer

    def se_block(se_block_layer_input, layer_n):
        se_block_layer = GlobalAveragePooling1D()(se_block_layer_input)
        se_block_layer = Dense(layer_n // 8, activation="relu")(se_block_layer)
        se_block_layer = Dense(layer_n, activation="sigmoid")(se_block_layer)
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
    large_kernel_size = 7
    small_kernel_size = 3
    depth = 1

    input_layer = Input(input_shape)  # 1024
    input_layer_1 = AveragePooling1D(8)(input_layer)  # 128

    # Encoder
    x = cbr(input_layer, layer_n, large_kernel_size, 1, 1)  # 1024
    for i in range(depth):
        x = resblock(x, layer_n, large_kernel_size, 1)
    out_0 = x

    x = cbr(x, layer_n * 2, small_kernel_size, 8, 1)  # 128
    for i in range(depth):
        x = resblock(x, layer_n * 2, small_kernel_size, 1)
    out_1 = x

    x = Concatenate()([x, input_layer_1])
    x = cbr(x, layer_n * 3, small_kernel_size, 8, 1)  # 16
    for i in range(depth):
        x = resblock(x, layer_n * 3, small_kernel_size, 1)

    # Decoder
    x = UpSampling1D(8)(x)
    x = Concatenate()([x, out_1])
    x = cbr(x, layer_n * 3, small_kernel_size, 1, 1)

    x = UpSampling1D(8)(x)
    x = Concatenate()([x, out_0])
    x = cbr(x, layer_n * 2, small_kernel_size, 1, 1)

    # binary classifier
    x = Conv1D(1, kernel_size=large_kernel_size, strides=1, padding="same")(x)
    out = Activation('sigmoid')(x)

    return Model(input_layer, out)
