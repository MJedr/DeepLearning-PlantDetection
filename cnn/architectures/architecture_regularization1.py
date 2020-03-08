from keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization, MaxPool1D
from keras import Sequential
from keras.regularizers import l1_l2, l1, l2
from keras.activations import relu, softmax
from keras.layers.advanced_activations import LeakyReLU


def architecture_CONV_FC_dropout(X, nbclasses, nb_conv=1,
                                 nb_fc=1, dropout_rate=0.5):
    '''
    tests only dropout rate and l1_l2 regularization
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-6
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                            'CONV_' + str(nb_fc) + '_FC128_dropout')
    model.add(Conv1D(input_shape=input_shape,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(Dropout(dropout_rate))

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l1_l2(l1_l2_rate),
                             activation=activation))
            model.add(Dropout(dropout_rate))

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    # SOFTMAX layer
    model.add(Dense(nbclasses, activation=softmax))

    # Create model.
    return model


def architecture_CONV_FC_batch_norm(X, nbclasses, nb_conv=1, nb_fc=1):
    '''
    tests batch norm with l1_l2 norm
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-6
    dropout_rate = 0.5
    conv_kernel = 9
    conv_filters = 64
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                            'CONV_' + 'k' + str(conv_kernel) +
                            str(nb_fc) + '_FC128_batch_norm')
    model.add(Conv1D(input_shape=input_shape,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l1_l2(l1_l2_rate),
                             activation=activation))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV_FC_batch_norm_dropout(X, nbclasses, nb_conv=1, nb_fc=1):
    '''
    dropout + batch norm + l1_l2
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-3
    dropout_rate = 0.8
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 236
    activation = relu

    model = Sequential(name=str(nb_conv) +
                            'CONV_' + 'k' + str(conv_kernel) +
                            str(nb_fc) +
                            '_FC128_bn_d' + str(dropout_rate))
    model.add(Conv1D(input_shape=input_shape,
                     activation=activation,
                     kernel_regularizer=l1_l2(l1_l2_rate),
                     kernel_initializer='he_normal',
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel,
                             activation=activation,
                             filters=conv_filters,
                             kernel_regularizer=l1_l2(l1_l2_rate)))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV_FC_batch_norm_dropout_l1_norm(X, nbclasses, nb_conv=1, nb_fc=1):
    '''
    dropout + batch norm + l1 norm
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_rate = 1.e-3
    dropout_rate = 0.8
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 236
    activation = relu

    model = Sequential(name=str(nb_conv) +
                            'CONV_' + 'k' + str(conv_kernel) +
                            str(nb_fc) +
                            '_FC128_bn_d' + str(dropout_rate) +
                            'L1_' + l1_rate)
    model.add(Conv1D(input_shape=input_shape,
                     activation=activation,
                     kernel_regularizer=l1(l1_rate),
                     kernel_initializer='he_normal',
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l1(l1_rate),
                             activation=activation))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1(l1_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV_FC_batch_norm_dropout_l1_norm(X, nbclasses, nb_conv=1, nb_fc=1):
    '''
    dropout + batch norm + l1 norm
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l2_rate = 1.e-3
    dropout_rate = 0.8
    conv_kernel = 9
    conv_filters = 64
    nbunits_fc = 236
    activation = relu

    model = Sequential(name=str(nb_conv) +
                            'CONV_' + 'k' + str(conv_kernel) +
                            str(nb_fc) +
                            '_FC128_bn_d' + str(dropout_rate) +
                            'L2_' + l2_rate)
    model.add(Conv1D(input_shape=input_shape,
                     kernel_regularizer=l2(l2_rate),
                     kernel_initializer='he_normal',
                     activation=activation,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l2(l2_rate)))
            model.add(LeakyReLU())
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l2(l2_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=relu))

    return model


def architecture_CONV_FC_batch_norm_dropout1(X, nbclasses, nb_conv=1, nb_fc=1):
    '''
    dropout + pooling + batch norm + l1_l2 norm
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-3
    dropout_rate = 0.8
    conv_kernel = 9
    conv_filters = 64
    nbunits_fc = 128
    activation = relu
    pool_kernel = 2

    model = Sequential(name=str(nb_conv) +
                            'CONV_' + 'k' + str(conv_kernel) +
                            'POOL' + str(pool_kernel) +
                            str(nb_fc) +
                            '_FC128_bn_d' + str(dropout_rate))
    model.add(Conv1D(input_shape=input_shape,
                     kernel_regularizer=l1_l2(l1_l2_rate),
                     kernel_initializer='he_normal',
                     activation=activation,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_kernel))

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             activation=activation,
                             kernel_regularizer=l1_l2(l1_l2_rate)))
            model.add(BatchNormalization())
            model.add(MaxPool1D(pool_kernel))

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc,
                        kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV_FC_batch_norm_dropout2(X, nbclasses, nb_conv=1, nb_fc=1):
    '''
    dropout + pooling + batch norm + l1_l2 norm; kernel size reduced to 3
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-3
    dropout_rate = 0.8
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu
    pool_kernel = 2

    model = Sequential(name=str(nb_conv) +
                            'CONV_' + 'k' + str(conv_kernel) +
                            'POOL' + str(pool_kernel) +
                            str(nb_fc) +
                            '_FC128_bn_d' + str(dropout_rate))
    model.add(Conv1D(input_shape=input_shape,
                     kernel_regularizer=l1_l2(l1_l2_rate),
                     kernel_initializer='he_normal',
                     activation=activation,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_kernel))

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             activation=activation,
                             kernel_regularizer=l1_l2(l1_l2_rate)))
            model.add(BatchNormalization())
            model.add(MaxPool1D(pool_kernel))

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc,
                        kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV9_FC_batch_norm_dropout_pool(X, nbclasses, nb_conv=1, nb_fc=1):
    '''
    dropout(strong; 0.8) + pooling + batch norm + l1_l2 norm
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-3
    dropout_rate = 0.5
    conv_kernel = 9
    conv_filters = 64
    pool_kernel = 3
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                            'CONV_' + str(nb_fc) +
                            'k' + str(conv_kernel) +
                            'f' + str(conv_filters) +
                            '_FC128_' +
                            'dropout' + str(dropout_rate) +
                            'batch_norm' + 'pool' +
                            str(pool_kernel)
                       )
    model.add(Conv1D(input_shape=input_shape,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(MaxPool1D(pool_size=pool_kernel, padding='same'))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l1_l2(l1_l2_rate),
                             activation=activation))
            model.add(MaxPool1D(pool_size=pool_kernel, padding='same'))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV3_FC_batch_norm_dropout_pool(X, nbclasses, nb_conv=1, nb_fc=1):
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-3
    dropout_rate = 0.5
    conv_kernel = 3
    conv_filters = 64
    pool_kernel = 3
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                            'CONV_' + str(nb_fc) +
                            'k' + str(conv_kernel) +
                            'f' + str(conv_filters) +
                            '_FC128_' +
                            'dropout' + str(dropout_rate) +
                            'batch_norm' + 'pool' +
                            str(pool_kernel)
                       )
    model.add(Conv1D(input_shape=input_shape,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(MaxPool1D(pool_size=pool_kernel, padding='same'))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l1_l2(l1_l2_rate),
                             activation=activation))
            model.add(MaxPool1D(pool_size=pool_kernel, padding='same'))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV3_FC256_batch_norm_dropout_pool(X, nbclasses, nb_conv=1, nb_fc=1,
                                                     dropout_rate=0.5):
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-3
    conv_kernel = 3
    conv_filters = 64
    pool_kernel = 5
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                            'CONV_' + str(nb_fc) +
                            'k' + str(conv_kernel) +
                            'f' + str(conv_filters) +
                            '_FC256_' +
                            'dropout' + str(dropout_rate) +
                            'batch_norm' + 'pool' +
                            str(pool_kernel)
                       )
    model.add(Conv1D(input_shape=input_shape,
                     activation=activation,
                     kernel_initializer='he_normal',
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(MaxPool1D(pool_size=pool_kernel, padding='same'))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l1_l2(l1_l2_rate),
                             activation=activation))
            model.add(MaxPool1D(pool_size=pool_kernel, padding='same'))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV3f32_FC256_batch_norm_dropout_pool(X, nbclasses, nb_conv=1, nb_fc=1):
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-3
    dropout_rate = 0.7
    conv_kernel = 3
    conv_filters = 32
    pool_kernel = 3
    nbunits_fc = 256
    activation = relu

    model = Sequential(name=str(nb_conv) +
                            'CONV_' + str(nb_fc) +
                            'k' + str(conv_kernel) +
                            'f' + str(conv_filters) +
                            '_FC256_' +
                            'dropout' + str(dropout_rate) +
                            'batch_norm' + 'pool' +
                            str(pool_kernel)
                       )
    model.add(Conv1D(input_shape=input_shape,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(MaxPool1D(pool_size=pool_kernel, padding='same'))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l1_l2(l1_l2_rate),
                             activation=activation))
            model.add(MaxPool1D(pool_size=pool_kernel, padding='same'))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV3f16_FC256_batch_norm_dropout_pool(X, nbclasses, nb_conv=1, nb_fc=1):
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-3
    dropout_rate = 0.7
    conv_kernel = 3
    conv_filters = 16
    pool_kernel = 3
    nbunits_fc = 256
    activation = relu

    model = Sequential(name=str(nb_conv) +
                            'CONV_' + str(nb_fc) +
                            'k' + str(conv_kernel) +
                            'f' + str(conv_filters) +
                            '_FC256_' +
                            'dropout' + str(dropout_rate) +
                            'batch_norm' + 'pool' +
                            str(pool_kernel)
                       )
    model.add(Conv1D(input_shape=input_shape,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(MaxPool1D(pool_size=pool_kernel, padding='same'))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l1_l2(l1_l2_rate),
                             activation=activation))
            model.add(MaxPool1D(pool_size=pool_kernel, padding='same'))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV3_FC_batch_norm_dropout_pool(X, nbclasses, nb_conv=1, nb_fc=1):
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-2
    dropout_rate = 0.5
    conv_kernel = 3
    conv_filters = 64
    pool_kernel = 3
    nbunits_fc = 256
    activation = relu

    model = Sequential(name=str(nb_conv) +
                            'CONV_' + str(nb_fc) +
                            'k' + str(conv_kernel) +
                            'f' + str(conv_filters) +
                            '_FC256_' +
                            'dropout' + str(dropout_rate) +
                            'batch_norm' + 'pool' +
                            'l_2_l1' + str(l1_l2_rate) +
                            str(pool_kernel)
                       )
    model.add(Conv1D(input_shape=input_shape,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(MaxPool1D(pool_size=pool_kernel, padding='same'))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l1_l2(l1_l2_rate),
                             activation=activation))
            model.add(MaxPool1D(pool_size=pool_kernel, padding='same'))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_basic(X, nbclasses, nb_conv=1, nb_fc=1,
                       dropout_rate=0.5, conv_filters=64,
                       pool_kernel=3,
                       conv_kernel=3, nb_neurons=256,
                       weight_initializer='he_nromal'):
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-3
    activation = relu

    model = Sequential(name=str(nb_conv) +
                            'CONV_' + str(nb_fc) +
                            'k' + str(conv_kernel) +
                            'f' + str(conv_filters) +
                            '_FC_' + str(nb_neurons) +
                            'dropout' + str(dropout_rate) +
                            'batch_norm' +
                            'pool' + str(pool_kernel))

    model.add(Conv1D(input_shape=input_shape,
                     activation=activation,
                     kernel_regularizer=l1_l2(l1_l2_rate),
                     kernel_initializer=weight_initializer,
                     kernel_size=conv_kernel,
                     filters=conv_filters))
    model.add(MaxPool1D(pool_size=pool_kernel, padding='same'))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv - 1):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l1_l2(l1_l2_rate),
                             activation=activation))
            model.add(MaxPool1D(pool_size=pool_kernel, padding='same'))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nb_neurons,
                        kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model
