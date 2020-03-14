from keras import Sequential
from keras.activations import relu, softmax, tanh, sigmoid
from keras.initializers import VarianceScaling
from keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l1_l2

vs = VarianceScaling(scale=1.0, mode="fan_in", distribution="normal", seed=None)


def architecture_CONV_FC_batch_norm_dropout_L1_l2(
    X, nbclasses, nb_conv=1, nb_fc=1, kernel_initializer="random_normal"
):
    """
    dropout + batch norm + l1_l2
    """
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.0e-3
    dropout_rate = 0.5
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = relu
    kernel_initializer = kernel_initializer

    model = Sequential(
        name=f"""{str(nb_conv)}__CONV_k{str(conv_kernel)}_
                              {str(nb_fc)}_initializer_{kernel_initializer}_
                               _FC128_bn_d_{str(dropout_rate)}"""
    )
    model.add(
        Conv1D(
            input_shape=input_shape,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l1_l2(l1_l2_rate),
            kernel_size=conv_kernel,
            filters=conv_filters,
        )
    )
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(
                Conv1D(
                    kernel_size=conv_kernel,
                    filters=conv_filters,
                    activation=activation,
                    kernel_regularizer=l1_l2(l1_l2_rate),
                )
            )
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(
            Dense(
                nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate), activation=activation
            )
        )
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV_FC_batch_norm_dropout_L1_l2_TANH(
    X, nbclasses, nb_conv=1, nb_fc=1
):
    """
    dropout + batch norm + l1_l2
    """
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.0e-3
    dropout_rate = 0.5
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = tanh

    model = Sequential(
        name=f"""{str(nb_conv)}_
                            CONV_k_{str(conv_kernel)}_
                            {str(nb_fc)}_FC128_bn_d_{str(dropout_rate)}
                            _TANH"""
    )
    model.add(
        Conv1D(
            input_shape=input_shape,
            activation=activation,
            kernel_regularizer=l1_l2(l1_l2_rate),
            kernel_size=conv_kernel,
            filters=conv_filters,
        )
    )
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(
                Conv1D(
                    kernel_size=conv_kernel,
                    filters=conv_filters,
                    activation=activation,
                    kernel_regularizer=l1_l2(l1_l2_rate),
                )
            )
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(
            Dense(
                nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate), activation=activation
            )
        )
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV_FC_batch_norm_dropout_L1_l2_SIGMOID(
    X, nbclasses, nb_conv=1, nb_fc=1
):
    """
    dropout + batch norm + l1_l2
    """
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.0e-3
    dropout_rate = 0.5
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128
    activation = sigmoid

    model = Sequential(
        name=f"""{str(nb_conv)}_CONV_k_
                      {str(conv_kernel)}_{str(nb_fc)}
                      _FC128_bn_d_{str(dropout_rate)}
                      _SIGMOID"""
    )
    model.add(
        Conv1D(
            input_shape=input_shape,
            activation=activation,
            kernel_regularizer=l1_l2(l1_l2_rate),
            kernel_size=conv_kernel,
            filters=conv_filters,
        )
    )
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(
                Conv1D(
                    kernel_size=conv_kernel,
                    filters=conv_filters,
                    activation=activation,
                    kernel_regularizer=l1_l2(l1_l2_rate),
                )
            )
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(
            Dense(
                nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate), activation=activation
            )
        )
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV_FC_batch_norm_dropout_L1_l2_LEAKY_ReLU(
    X, nbclasses, nb_conv=1, nb_fc=1
):
    """
    dropout + batch norm + l1_l2
    """
    # input size
    width, height, depth = X.shape
    input_shape = (height, depth)

    # parameters of the architecture
    l1_l2_rate = 1.0e-3
    dropout_rate = 0.5
    conv_kernel = 3
    conv_filters = 64
    nbunits_fc = 128

    model = Sequential(
        name=f"""{str(nb_conv)}_CONV_k_{str(conv_kernel)}_
                            {str(nb_fc)}_FC128_bn_d_{str(dropout_rate)}
                            _LEAKY_ReLU"""
    )
    model.add(
        Conv1D(
            input_shape=input_shape,
            kernel_regularizer=l1_l2(l1_l2_rate),
            kernel_size=conv_kernel,
            filters=conv_filters,
        )
    )
    model.add(LeakyReLU())
    model.add(BatchNormalization())

    # if more covolutional layers are defined in parameters
    if nb_conv > 1:
        for _layer in range(nb_conv):
            model.add(
                Conv1D(
                    kernel_size=conv_kernel,
                    filters=conv_filters,
                    kernel_regularizer=l1_l2(l1_l2_rate),
                )
            )
            model.add(LeakyReLU())
            model.add(BatchNormalization())

    # Flatten + FC layers
    model.add(Flatten())
    for _layer in range(nb_fc):
        model.add(Dense(nbunits_fc, kernel_regularizer=l1_l2(l1_l2_rate)))
        model.add(LeakyReLU())
        model.add(Dropout(dropout_rate))

    model.add(Dense(nbclasses, activation=softmax))

    return model


def architecture_CONV_FC_batch_norm_dropout_L1_l2_XAVIER(
    X, nbclasses, nb_conv=1, nb_fc=1
):
    return architecture_CONV_FC_batch_norm_dropout_L1_l2(
        X, nbclasses, nb_conv=1, nb_fc=1, kernel_initializer="glorot_normal"
    )


def architecture_CONV_FC_batch_norm_dropout_L1_l2_V_SCALLING(
    X, nbclasses, nb_conv=1, nb_fc=1
):
    return architecture_CONV_FC_batch_norm_dropout_L1_l2(
        X, nbclasses, nb_conv=1, nb_fc=1, kernel_initializer=vs
    )


def architecture_CONV_FC_batch_norm_dropout_L1_l2_H_NORMAL(
    X, nbclasses, nb_conv=1, nb_fc=1
):
    return architecture_CONV_FC_batch_norm_dropout_L1_l2(
        X, nbclasses, nb_conv=1, nb_fc=1, kernel_initializer="he_normal"
    )


def architecture_CONV_FC_batch_norm_dropout_L1_l2_H_UNIFORM(
    X, nbclasses, nb_conv=1, nb_fc=1
):
    return architecture_CONV_FC_batch_norm_dropout_L1_l2(
        X, nbclasses, nb_conv=1, nb_fc=1, kernel_initializer="lecun_normal"
    )
