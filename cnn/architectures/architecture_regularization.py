from keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization, MaxPool1D
from keras import Sequential
from keras.regularizers import l1_l2, l1, l2
from keras.activations import relu, softmax
from keras.layers.advanced_activations import LeakyReLU


# dropout without additional regularization
def architecture_CONV_FC_dropout(X, nbclasses, nb_conv=1,
                                 nb_fc=1, dropout_rate=0.5):
    '''
   tests only dropout rate and l1_l2 regularization
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                       'CONV_' + str(nb_fc) + '_FC128_dropout')
    model.add(Conv1D(input_shape=input_shape,
                     activation=activation, 
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(Dropout(dropout_rate))

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             activation=activation))
            model.add(Dropout(dropout_rate))

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc,
                        activation=activation))
        model.add(Dropout(dropout_rate))

    # SOFTMAX layer
    model.add(Dense(nbclasses, activation=softmax))

    # Create model.
    return model

# dropout with elastic search regularization
def architecture_CONV_FC_dropout_L1_L2(X, nbclasses, nb_conv=1,
                                 nb_fc=1, dropout_rate=0.5):
    '''
   tests only dropout rate and l1_l2 regularization
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-3
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                       'CONV_' + str(nb_fc) + '_FC128_dropout'+
                       'L1_l2_R')
    model.add(Conv1D(input_shape=input_shape,
                     activation=activation,
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

# dropout with L1 regularization
def architecture_CONV_FC_dropout_L1(X, nbclasses, nb_conv=1,
                                 nb_fc=1, dropout_rate=0.5):
    '''
   tests only dropout rate and l1_l2 regularization
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_rate = 1.e-3
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                       'CONV_' + str(nb_fc) + '_FC128_dropout' +
                       'L1R')
    model.add(Conv1D(input_shape=input_shape,
                     activation=activation,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(Dropout(dropout_rate))

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l1(l1_rate),
                             activation=activation))
            model.add(Dropout(dropout_rate))

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1(l1_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    # SOFTMAX layer
    model.add(Dense(nbclasses, activation=softmax))

    # Create model.
    return model


# dropout with L2 regularization
def architecture_CONV_FC_dropout_L2(X, nbclasses, nb_conv=1,
                                 nb_fc=1, dropout_rate=0.5):
    '''
   tests only dropout rate and l1_l2 regularization
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l2_rate = 1.e-3
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                       'CONV_' + str(nb_fc) + '_FC128_dropout' +
                       'L2R')
    model.add(Conv1D(input_shape=input_shape,
                     activation=activation,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(Dropout(dropout_rate))

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l2(l2_rate),
                             activation=activation))
            model.add(Dropout(dropout_rate))

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l2(l2_rate),
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
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                       'CONV_' + 'k' + str(conv_kernel) +
                        str(nb_fc) + '_FC128_batch_norm')
    model.add(Conv1D(input_shape=input_shape,
                     activation=activation,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             activation=activation))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc,
                        activation=activation))

    model.add(Dense(nbclasses, activation=softmax))

    return model


# batch norm L1 - L2
def architecture_CONV_FC_batch_norm_L1_L2(X, nbclasses, nb_conv=1, nb_fc=1):
    '''
    tests batch norm with l1_l2 norm
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu
    l1_l2_rate = 1.e-3

    model = Sequential(name=str(nb_conv) +
                       'CONV_' + 'k' + str(conv_kernel) +
                        str(nb_fc) + '_FC128_batch_norm' +
                       'L1_L2_'+str(l1_l2_rate))
    model.add(Conv1D(input_shape=input_shape,
                     activation=activation,
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

# batch norm L1
def architecture_CONV_FC_batch_norm_L1(X, nbclasses, nb_conv=1, nb_fc=1):
    '''
    tests batch norm with L1 norm
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu
    l1_rate = 1.e-3

    model = Sequential(name=str(nb_conv) +
                       'CONV_' + 'k' + str(conv_kernel) +
                        str(nb_fc) + '_FC128_batch_norm' +
                       'L1_'+ str(l1_rate))
    model.add(Conv1D(input_shape=input_shape,
                     activation=activation,
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

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV_FC_batch_norm_L2(X, nbclasses, nb_conv=1, nb_fc=1):
    '''
    tests batch norm with L2 norm
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu
    l2_rate = 1.e-3

    model = Sequential(name=str(nb_conv) +
                       'CONV_' + 'k' + str(conv_kernel) +
                        str(nb_fc) + '_FC128_batch_norm' +
                       'L2_'+ str(l2_rate))
    model.add(Conv1D(input_shape=input_shape,
                     activation=activation,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l2(l2_rate),
                             activation=activation))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l2(l2_rate),
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
    dropout_rate = 0.5
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                       'CONV_' + 'k' + str(conv_kernel) +
                        str(nb_fc) +
                        '_FC128_bn_d' + str(dropout_rate))
    model.add(Conv1D(input_shape=input_shape,
                     activation=activation,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             activation=activation))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc,
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV_FC_batch_norm_dropout_L1_l2(X, nbclasses, nb_conv=1, nb_fc=1):
    '''
    dropout + batch norm + l1_l2
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-3
    dropout_rate = 0.5
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                       'CONV_' + 'k' + str(conv_kernel) +
                        str(nb_fc) +
                        '_FC128_bn_d' + str(dropout_rate))
    model.add(Conv1D(input_shape=input_shape,
                     activation=activation,
                     kernel_regularizer=l1_l2(l1_l2_rate),
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             activation=activation,
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

def architecture_CONV_FC_batch_norm_dropout_L1(X, nbclasses, nb_conv=1, nb_fc=1):
    '''
    dropout + batch norm + l1_l2
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_rate = 1.e-3
    dropout_rate = 0.5
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                       'CONV_' + 'k' + str(conv_kernel) +
                        str(nb_fc) +
                        '_FC128_bn_d' + str(dropout_rate))
    model.add(Conv1D(input_shape=input_shape,
                     kernel_regularizer=l1(l1_rate),
                     activation=activation,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             activation=activation,
                             kernel_regularizer=l1(l1_rate)))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1(l1_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model

def architecture_CONV_FC_batch_norm_dropout_L2(X, nbclasses, nb_conv=1, nb_fc=1):
    '''
    dropout + batch norm + l1_l2
    '''
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_rate = 1.e-3
    dropout_rate = 0.5
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                       'CONV_' + 'k' + str(conv_kernel) +
                        str(nb_fc) +
                        '_FC128_bn_d' + str(dropout_rate))
    model.add(Conv1D(input_shape=input_shape,
                     kernel_regularizer=l2(l2_rate),
                     activation=activation,
                     kernel_size=conv_kernel, filters=conv_filters))
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             activation=activation,
                             kernel_regularizer=l2(l2_rate)))
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l2(l2_rate),
                        activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model