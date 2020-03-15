from keras import Sequential
from keras.activations import relu
from keras.layers import BatchNormalization
from keras.layers import Flatten, Dense, Dropout, Conv1D, MaxPool1D
from keras.layers import LSTM, CuDNNLSTM
from keras.regularizers import l1_l2


def architecture_rnn_dropout(X_tr, nb_LSTM=1, units_LSTM=256, fc=1, nb_units_fc=256):
    width, height, depth = X_tr.shape
    input_shape = (height, depth)

    model = Sequential(
        name="RNN_dropout"
        + str(nb_LSTM)
        + "_"
        + str(units_LSTM)
        + "fc_"
        + str(fc)
        + "_"
        + str(nb_units_fc)
        + "_dropout0.5"
    )

    model.add(CuDNNLSTM(units_LSTM, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.5))

    if nb_LSTM > 1:
        for lstm in range(nb_LSTM - 1):
            model.add(CuDNNLSTM(units_LSTM, return_sequences=True))
            model.add(Dropout(0.5))

    model.add(Flatten())

    for dense in range(fc):
        model.add(Dense(nb_units_fc, activation="relu"))
        model.add(Dropout(0.5))

    model.add(Dense(9, activation="softmax"))
    return model


def architecture_rnn_cascade(X_tr, nb_LSTM=1, units_LSTM=256, fc=1, nb_units_fc=256):
    width, height, depth = X_tr.shape
    input_shape = (height, depth)

    model = Sequential(
        name="RNN_cascade_"
        + str(nb_LSTM)
        + "_"
        + str(units_LSTM)
        + "fc_"
        + str(fc)
        + "_"
        + str(nb_units_fc)
        + "_dropout0.5"
    )
    model.add(CuDNNLSTM(units_LSTM, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.5))

    if nb_LSTM > 1:
        for lstm in range(nb_LSTM - 1):
            units_LSTM = units_LSTM / 2
            model.add(CuDNNLSTM(int(units_LSTM), return_sequences=True))
            model.add(Dropout(0.5))

    model.add(Flatten())

    for dense in range(fc):
        model.add(Dense(nb_units_fc, activation="relu"))
        model.add(Dropout(0.5))

    model.add(Dense(9, activation="softmax"))
    return model


def architecture_rnn_cascade_regularization(
    X_tr, nb_LSTM=1, units_LSTM=256, fc=1, nb_units_fc=256, activation=relu
):
    width, height, depth = X_tr.shape
    input_shape = (height, depth)

    model = Sequential(
        name="RNN_cascade_"
        + str(nb_LSTM)
        + "_"
        + str(units_LSTM)
        + "fc_"
        + str(fc)
        + "_"
        + str(nb_units_fc)
        + "_dropout0.5"
    )
    model.add(
        CuDNNLSTM(
            units_LSTM,
            input_shape=input_shape,
            return_sequences=True,
            kernel_regularizer=l1_l2(0.001, 0.001),
        )
    )
    model.add(Dropout(0.5))

    if nb_LSTM > 1:
        for _lstm in range(nb_LSTM - 1):
            units_LSTM = units_LSTM / 2
            model.add(
                CuDNNLSTM(
                    int(units_LSTM),
                    return_sequences=True,
                    kernel_regularizer=l1_l2(0.001, 0.001),
                )
            )
            model.add(Dropout(0.5))

    model.add(Flatten())

    for _dense in range(fc):
        model.add(
            Dense(
                nb_units_fc, activation="relu", kernel_regularizer=l1_l2(0.001, 0.001)
            )
        )
        model.add(Dropout(0.5))

    model.add(Dense(9, activation="softmax"))
    return model


def architecture_rnn_cascade_regularization_bn(
    X_tr, nb_LSTM=1, units_LSTM=256, fc=1, nb_units_fc=256
):
    width, height, depth = X_tr.shape
    input_shape = (height, depth)

    model = Sequential(
        name="RNN_cascade_"
        + str(nb_LSTM)
        + "_"
        + str(units_LSTM)
        + "fc_"
        + str(fc)
        + "_"
        + str(nb_units_fc)
        + "l1_l2"
        + "_bn"
    )
    model.add(
        CuDNNLSTM(
            units_LSTM,
            input_shape=input_shape,
            return_sequences=True,
            kernel_regularizer=l1_l2(0.001, 0.001),
        )
    )
    model.add(BatchNormalization())

    if nb_LSTM > 1:
        for lstm in range(nb_LSTM - 1):
            units_LSTM = units_LSTM / 2
            model.add(
                CuDNNLSTM(
                    int(units_LSTM),
                    return_sequences=True,
                    kernel_regularizer=l1_l2(0.001, 0.001),
                )
            )
            model.add(BatchNormalization())

    model.add(Flatten())

    for dense in range(fc):
        model.add(
            Dense(
                nb_units_fc, activation="relu", kernel_regularizer=l1_l2(0.001, 0.001)
            )
        )
        model.add(Dropout(0.5))

    model.add(Dense(9, activation="softmax"))
    return model


def architecture_rnn_cascade_regularization_bn(
    X_tr, nb_LSTM=1, units_LSTM=256, fc=1, nb_units_fc=256
):
    width, height, depth = X_tr.shape
    input_shape = (height, depth)

    model = Sequential(
        name="RNN_cascade_"
        + str(nb_LSTM)
        + "_"
        + str(units_LSTM)
        + "fc_"
        + str(fc)
        + "_"
        + str(nb_units_fc)
        + "l1_l2"
        + "_bn"
    )
    model.add(
        CuDNNLSTM(
            units_LSTM,
            input_shape=input_shape,
            return_sequences=True,
            kernel_regularizer=l1_l2(0.001, 0.001),
        )
    )
    model.add(BatchNormalization())

    if nb_LSTM > 1:
        for lstm in range(nb_LSTM - 1):
            units_LSTM = units_LSTM / 2
            model.add(
                CuDNNLSTM(
                    int(units_LSTM),
                    return_sequences=True,
                    kernel_regularizer=l1_l2(0.001, 0.001),
                )
            )
            model.add(BatchNormalization())

    model.add(Flatten())

    for dense in range(fc):
        model.add(
            Dense(
                nb_units_fc, activation="relu", kernel_regularizer=l1_l2(0.001, 0.001)
            )
        )
        model.add(Dropout(0.5))

    model.add(Dense(9, activation="softmax"))
    return model


def architecture_rnn_batch_norm(X_tr, nb_LSTM=1, units_LSTM=256, fc=1, nb_units_fc=256):
    width, height, depth = X_tr.shape
    input_shape = (height, depth)

    model = Sequential(
        name="RNN_batch_norm"
        + str(nb_LSTM)
        + "_"
        + str(units_LSTM)
        + "fc_"
        + str(fc)
        + "_"
        + str(nb_units_fc)
        + "batchnorm"
    )
    model.add(CuDNNLSTM(units_LSTM, input_shape=input_shape, return_sequences=True))
    model.add(BatchNormalization())

    if nb_LSTM > 1:
        for lstm in range(nb_LSTM - 1):
            model.add(
                CuDNNLSTM(units_LSTM, input_shape=input_shape, return_sequences=True)
            )
            model.add(BatchNormalization())

    model.add(Flatten())

    for dense in range(fc):
        model.add(Dense(nb_units_fc, activation="relu"))
        model.add(Dropout(0.5))

    model.add(Dense(9, activation="softmax"))
    return model


def architecture_rnn_batch_norm_L1_l2(
    X_tr, nb_LSTM=1, units_LSTM=256, fc=1, nb_units_fc=256
):
    width, height, depth = X_tr.shape
    input_shape = (height, depth)

    model = Sequential(
        name="RNN__batch_norm_L1_l2"
        + str(nb_LSTM)
        + "_"
        + str(units_LSTM)
        + "fc_"
        + str(fc)
        + "_"
        + str(nb_units_fc)
        + "_l1_l2_bn"
    )
    model.add(
        CuDNNLSTM(
            units_LSTM,
            input_shape=input_shape,
            return_sequences=True,
            kernel_regularizer=l1_l2(0.001),
            kernel_initializer="he_normal",
        )
    )
    model.add(BatchNormalization())

    if nb_LSTM > 1:
        for lstm in range(nb_LSTM - 1):
            model.add(
                CuDNNLSTM(
                    units_LSTM,
                    input_shape=input_shape,
                    return_sequences=True,
                    kernel_regularizer=l1_l2(0.001, 0.001),
                )
            )
            model.add(BatchNormalization())

    model.add(Flatten())

    for dense in range(fc):
        model.add(Dense(nb_units_fc, activation="relu"))
        model.add(Dropout(0.5))

    model.add(Dense(9, activation="softmax"))
    return model


def architecture_rnn_cascade_regularization_bn_dropout(
    X_tr, nb_LSTM=1, units_LSTM=256, fc=1, nb_units_fc=256, activation=relu, dr=0.5
):
    width, height, depth = X_tr.shape
    input_shape = (height, depth)

    model = Sequential(
        name="RNN_cascade_"
        + str(nb_LSTM)
        + "_"
        + str(units_LSTM)
        + "fc_"
        + str(fc)
        + "_"
        + str(nb_units_fc)
        + "l1_l2"
        + "_bn_dropoutLSTM"
        + str(dr)
    )
    model.add(
        LSTM(
            units_LSTM,
            input_shape=input_shape,
            return_sequences=True,
            dropout=dr,
            kernel_regularizer=l1_l2(0.001, 0.001),
        )
    )
    model.add(BatchNormalization())

    if nb_LSTM > 1:
        for lstm in range(nb_LSTM - 1):
            units_LSTM = units_LSTM / 2
            model.add(
                LSTM(
                    int(units_LSTM),
                    return_sequences=True,
                    dropout=dr,
                    kernel_regularizer=l1_l2(0.001, 0.001),
                )
            )
            model.add(BatchNormalization())

    model.add(Flatten())

    for dense in range(fc):
        model.add(
            Dense(
                nb_units_fc, activation="relu", kernel_regularizer=l1_l2(0.001, 0.001)
            )
        )
        model.add(Dropout(0.5))

    model.add(Dense(9, activation="softmax"))
    return model


def architecture_rnn_cascade_recurrent_regularization_bn_(
    X_tr, nb_LSTM=1, units_LSTM=256, fc=1, nb_units_fc=256, dr=0.5
):
    width, height, depth = X_tr.shape
    input_shape = (height, depth)

    model = Sequential(
        name="RNN_cascade_"
        + str(nb_LSTM)
        + "_"
        + str(units_LSTM)
        + "fc_"
        + str(fc)
        + "_"
        + str(nb_units_fc)
        + "dr_normal"
        + str(dr)
        + "recurrent_l1_l2"
        + "_bn"
    )
    model.add(
        CuDNNLSTM(
            units_LSTM,
            input_shape=input_shape,
            return_sequences=True,
            recurrent_regularizer=l1_l2(0.001, 0.001),
            kernel_regularizer=l1_l2(0.001, 0.001),
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(dr))

    if nb_LSTM > 1:
        for lstm in range(nb_LSTM - 1):
            units_LSTM = units_LSTM / 2
            model.add(
                CuDNNLSTM(
                    int(units_LSTM),
                    return_sequences=True,
                    recurrent_regularizer=l1_l2(0.001, 0.001),
                    kernel_regularizer=l1_l2(0.001, 0.001),
                )
            )
            model.add(BatchNormalization())
            model.add(Dropout(dr))

    model.add(Flatten())

    for dense in range(fc):
        model.add(
            Dense(
                nb_units_fc, activation="relu", kernel_regularizer=l1_l2(0.001, 0.001)
            )
        )
        model.add(Dropout(0.5))

    model.add(Dense(9, activation="softmax"))
    return model
