import csv
import datetime
import os

import keras
import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Input, Dense, Flatten, Dropout, Conv1D, UpSampling1D, \
    MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def reduceLR(patience=5, min_lr=0.000001):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6,
                                  patience=2, min_lr=min_lr)
    return reduce_lr


def preprocess_data(dataset, x_col_name, y_col_name, polygon_ind_col):
    X = dataset[x_col_name].apply(lambda x: np.array(x)).values
    X = np.concatenate(X).reshape(X.shape[0], X[1].shape[0], 1)
    X = (X) / (np.max(X))
    print(X.shape)
    # split dataset according to polygon numbers -
    # we don't want samples from same polygon in tain and test set
    unique_indexes = dataset.drop_duplicates(polygon_ind_col)
    _X_tr, _X_te, y_tr, y_te = train_test_split(unique_indexes[polygon_ind_col], unique_indexes[y_col_name])
    X_tr = X[dataset[dataset.indeks.isin(_X_tr.values)].index]
    y_tr = dataset[dataset.indeks.isin(_X_tr.values)][y_col_name]
    X_te = X[dataset[dataset.indeks.isin(_X_te.values)].index]
    y_te = dataset[dataset.indeks.isin(_X_te.values)][y_col_name]
    return X_tr, X_te, y_tr, y_te


def train_model(model, X_tr, X_te, y_tr, y_te):
    print(X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_tr, y_tr, batch_size=256, epochs=200,
                        shuffle=True, validation_data=(X_te, y_te),
                        callbacks=[reduce_lr])
    return (model, history)


def autoencoder():
    input_img = Input(shape=(430, 1))
    x = Conv1D(16, 5, activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 5, activation='relu', padding='same')(x)
    encoded = BatchNormalization()(x)

    x = Conv1D(16, 5, activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 1, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder


def encoder(input_img):
    x = Conv1D(16, 5, activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 5, activation='relu', padding='same')(x)
    encoded = BatchNormalization()(x)
    return encoded


def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    den1 = Dropout(0.5)(den)
    out = Dense(num_classes, activation='softmax')(den1)
    return out


def train_autoencoder(X_tr, X_te, save_weights=False):
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(X_tr, X_tr,
                    epochs=10,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(X_te, X_te))
    if save_weights:
        autoencoder.save_weights('autoencoder.h5')
    return [layer.get_weights() for layer in autoencoder.layers[0:8]]


def one_hot_encode_y(y_tr, y_te):
    train_Y_one_hot = to_categorical(y_tr)
    test_Y_one_hot = to_categorical(y_te)
    return train_Y_one_hot, test_Y_one_hot


def create_logging_file(name=r'outputs/experiment_results.csv'):
    fields = ['model_name',
              'F_u_f1', 'F_u_PA', 'F_u_UA',
              'M_c_f1', 'M_c_PA', 'M_c_UA',
              'Other_f1', 'Other_PA', 'Other_UA',
              'OA', 'OA_WER', 'LOSS', 'LOSS_WER']

    with open(name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        csvfile.close()

    return


if __name__ == "__main__":
    reduce_lr = reduceLR()
    adam = Adam(lr=0.01, clipnorm=1.)

    dataset = pd.read_pickle(r'..\..\outputs\dataset_train_k5.pickle')

    if len(dataset.columns) > 5:
        dataset = dataset.iloc[:, 1:]

    dataset.columns = ['x', 'y', 'ekstrakcja', 'klasa', 'indeks']

    classes = {'Cienie': 0, 'Drogi': 1,
               'Drzewa iglaste': 2, 'Drzewa lisciaste': 3,
               'Dzewa iglaste': 2, 'Fil_ulm': 4,
               'Mol_cae': 5, 'Pola uprawne': 6,
               'X_niegatunek': 7, 'Zabudowa': 8}

    dataset['klasa_id'] = dataset.klasa.map(classes)

    X_tr, X_te, y_tr, y_te = preprocess_data(dataset, 'ekstrakcja', 'klasa_id', 'indeks')
    autoencoder = autoencoder()
    weights = train_autoencoder(X_tr, X_te, False)
    train_Y_one_hot, test_Y_one_hot = one_hot_encode_y(y_tr, y_te)
    train_X, valid_X, train_label, valid_label = train_test_split(X_tr, train_Y_one_hot,
                                                                  test_size=0.2, random_state=13)

    encode = encoder(input_img)
    full_model = Model(input_img, fc(encode), name='AutoencoderCNN')

    for l1, l2 in zip(full_model.layers[:5], weights):
        l1.set_weights(l2)

    for layer in full_model.layers[0:5]:
        layer.trainable = False

    logging_file = 'autoencoder.csv'
    if not os.path.isfile(logging_file):
        create_logging_file(logging_file)

    log_dir = "logs//fit//" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    full_model.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=keras.optimizers.Adam(),
                       metrics=['accuracy'])
    classify_train = full_model.fit(X_tr, train_Y_one_hot, batch_size=256,
                                    epochs=100, verbose=1,
                                    validation_data=(X_te, test_Y_one_hot),
                                    callbacks=[tensorboard_callback])

    y_pred = full_model.predict(X_te)
    y_pred = predicted_classes = np.argmax(np.round(y_pred), axis=1)
