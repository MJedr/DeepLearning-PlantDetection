# The purpose of the script is to test the crucial parameters
# of the convolutional network - the number of convolutional
# layers, number of fully connected layers and the convolutional
# kernel size

from keras.layers import Conv1D, Flatten, Dense
from keras import Sequential
from keras.regularizers import l1_l2
from keras.activations import relu, softmax


def architecture_CONV_k9_f64_FC_basic(X, nbclasses, nb_conv=1, nb_fc=1):
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-6
    conv_kernel = 9
    conv_filters = 64
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) + 'CONV_k9_f64' +
                       str(nb_fc) + '_FC128_basic')
    model.add(Conv1D(input_shape=input_shape,
              kernel_size=conv_kernel, filters=conv_filters))

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv-1):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l1_l2(l1_l2_rate),
                             activation=activation))
    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))

    # Softmax layer
    model.add(Dense(nbclasses, activation=softmax))

    # Create model
    return model


def architecture_CONV_k3_f64_FC_basic(X, nbclasses, nb_conv=1, nb_fc=1):
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-6
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) + '_CONV_k12_f64_' +
                       str(nb_fc) + '_FC128_basic')
    model.add(Conv1D(input_shape=input_shape,
              kernel_size=conv_kernel, filters=conv_filters))

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv-1):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l1_l2(l1_l2_rate),
                             activation=activation))
    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))

    # Softmax layer
    model.add(Dense(nbclasses, activation=softmax))

    # Create model
    return model


def architecture_CONV_k12_f64_FC_basic(X, nbclasses, nb_conv=1, nb_fc=1):
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.e-6
    conv_kernel = 12
    conv_filters = 64
    nbunits_fc = 128
    activation = relu

    model = Sequential(name=str(nb_conv) +
                       '_CONV_k12_f64_' + str(nb_fc) + '_FC128_basic')
    model.add(Conv1D(input_shape=input_shape,
              kernel_size=conv_kernel, filters=conv_filters))

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for add in range(nb_conv-1):
            model.add(Conv1D(kernel_size=conv_kernel, filters=conv_filters,
                             kernel_regularizer=l1_l2(l1_l2_rate),
                             activation=activation))
    # Flatten + FC layers
    model.add(Flatten())
    for add in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate),
                        activation=activation))

    # Softmax layer
    model.add(Dense(nbclasses, activation=softmax))

    # Create model
    return model
