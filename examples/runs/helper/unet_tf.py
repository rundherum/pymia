import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, layer_idx, filters_root, kernel_size, dropout_rate, padding, activation, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.layer_idx = layer_idx
        self.filters_root = filters_root
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.activation = activation

        filters = _get_filter_count(layer_idx, filters_root)
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=filters,
                                               kernel_size=(kernel_size, kernel_size),
                                               strides=1,
                                               padding=padding)
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.activation_1 = tf.keras.layers.Activation(activation)

        self.conv2d_2 = tf.keras.layers.Conv2D(filters=filters,
                                               kernel_size=(kernel_size, kernel_size),
                                               strides=1,
                                               padding=padding)
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.activation_2 = tf.keras.layers.Activation(activation)

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        x = self.conv2d_1(x)

        if training:
            x = self.dropout_1(x)
        x = self.activation_1(x)
        x = self.conv2d_2(x)

        if training:
            x = self.dropout_2(x)

        x = self.activation_2(x)
        return x


class UpconvBlock(tf.keras.layers.Layer):

    def __init__(self, layer_idx, filters_root, kernel_size, pool_size, padding, activation, **kwargs):
        super(UpconvBlock, self).__init__(**kwargs)
        self.layer_idx = layer_idx
        self.filters_root = filters_root
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding
        self.activation = activation

        filters = _get_filter_count(layer_idx + 1, filters_root)
        self.upconv = tf.keras.layers.Conv2DTranspose(filters // 2,
                                                      kernel_size=(pool_size, pool_size),
                                                      strides=pool_size, padding=padding)

        self.activation_1 = tf.keras.layers.Activation(activation)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.upconv(x)
        x = self.activation_1(x)
        return x


class CropConcatBlock(tf.keras.layers.Layer):

    def call(self, x, skip_x, **kwargs):
        skip_shape = tf.shape(skip_x)
        up_shape = tf.shape(x)

        height_diff = (skip_shape[1] - up_shape[1])
        width_diff = (skip_shape[2] - up_shape[2])

        height_pad = [height_diff // 2, height_diff // 2 + (height_diff % 2)]
        width_pad = [width_diff // 2, width_diff // 2 + (width_diff % 2)]

        if height_diff > 0 or width_diff > 0:
            x = tf.pad(x, tf.convert_to_tensor([[0, 0], height_pad, width_pad, [0, 0]]))

        x = tf.concat([skip_x, x], axis=-1)
        return x


def _get_filter_count(layer_idx, filters_root):
    return 2 ** layer_idx * filters_root


def build_model(nx = None, ny = None, channels: int = 1, num_classes: int = 2,
                layer_depth: int = 5, filters_root: int = 64, kernel_size: int = 3, pool_size: int = 2,
                dropout_rate: int = 0.0, padding: str = 'same', activation='relu') -> tf.keras.Model:

    inputs = tf.keras.Input(shape=(nx, ny, channels), name='inputs')

    x = inputs
    contracting_layers = {}

    conv_params = dict(filters_root=filters_root, kernel_size=kernel_size, dropout_rate=dropout_rate, padding=padding,
                       activation=activation)

    for layer_idx in range(0, layer_depth - 1):
        x = ConvBlock(layer_idx, **conv_params)(x)
        contracting_layers[layer_idx] = x
        x = tf.keras.layers.MaxPooling2D((pool_size, pool_size))(x)

    x = ConvBlock(layer_idx + 1, **conv_params)(x)

    for layer_idx in range(layer_idx, -1, -1):
        x = UpconvBlock(layer_idx,
                        filters_root,
                        kernel_size,
                        pool_size,
                        padding,
                        activation)(x)
        x = CropConcatBlock()(x, contracting_layers[layer_idx])
        x = ConvBlock(layer_idx, **conv_params)(x)

    x = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=1, padding=padding)(x)

    outputs = tf.keras.layers.Activation(None, name='outputs')(x)
    model = tf.keras.Model(inputs, outputs, name='unet')

    return model