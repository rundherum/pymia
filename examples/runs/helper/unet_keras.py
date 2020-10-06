import tensorflow.keras.models as models
import tensorflow.keras.backend as back
import tensorflow.keras.layers as layers


def conv2d_block(inputs, use_batch_norm=True, dropout=0.3, filters=16, kernel_size=(3, 3), activation="relu",
                 kernel_initializer="he_normal", padding="same"):
    c = layers.Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding,
                      use_bias=not use_batch_norm)(inputs)
    if use_batch_norm:
        c = layers.BatchNormalization()(c)
    if dropout > 0.0:
        c = layers.Dropout(dropout)(c)
    c = layers.Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding,
                      use_bias=not use_batch_norm)(c)
    if use_batch_norm:
        c = layers.BatchNormalization()(c)
    return c


def get_unet(input_shape, num_classes=1, dropout=0.0, filters=32, num_layers=4, output_activation='sigmoid'):  # 'sigmoid' or 'softmax'

    # Build U-Net model
    inputs = layers.Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='valid')
        down_layers.append(x)
        x = layers.MaxPooling2D((2, 2), strides=2)(x)
        filters = filters * 2  # double the number of filters with each layer

    x = layers.Dropout(dropout)(x)
    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='valid')

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='valid')(x)

        ch, cw = get_crop_shape(back.int_shape(conv), back.int_shape(x))
        conv = layers.Cropping2D(cropping=(ch, cw))(conv)

        x = layers.concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='valid')

    outputs = layers.Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = target[2] - refer[2]
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = target[1] - refer[1]
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)
