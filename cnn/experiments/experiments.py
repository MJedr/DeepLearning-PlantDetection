import csv
import json
import os
import random

import numpy as np
from keras.optimizers import adam, Adadelta, SGD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from cnn.architectures import architecture_nconv
from cnn.architectures import architecture_pool, architecture_activation_and_initializers, architecture_regularization
from cnn_callbacks import reduceLR, earlyStop

# callbacks
reduce_lr = reduceLR()
early_stop = earlyStop()
adam = adam(lr=0.0001, clipnorm=1.)
sgd = SGD()
adadelta = Adadelta(lr=1.0, rho=0.95)
sgd = SGD(lr=0.0001, momentum=0.0, nesterov=False)


def get_model_params(param_dict):
    architecture = {}
    for parameter, param_vals in param_dict.items():
        architecture[parameter] = random.choice(param_vals)
    return architecture


def save_architecture(architecture, name):
    with open(name, 'w') as file:
        json.dump(architecture, file)


def create_logging_file(name=r'../results/experiment_results.csv'):
    if os.path.isdir(r'../results'):
        os.mkdir(r'../results')
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


def preprocess_data(dataset, x_col_name, y_col_name, polygon_ind_col):
    X = dataset[x_col_name].apply(lambda x: np.array(x)).values
    X = np.concatenate(X).reshape(X.shape[0], X[1].shape[0], 1)
    # X = (X - np.min(X))/(np.max(X) - np.min(X))
    # split dataset according to polygon numbers -
    # we don't want samples from same polygon in tain and test set
    unique_indexes = dataset.drop_duplicates(polygon_ind_col)
    _X_tr, _X_te, y_tr, y_te = train_test_split(unique_indexes[polygon_ind_col], unique_indexes[y_col_name])
    X_tr = X[dataset[dataset.indeks.isin(_X_tr.values)].index]
    y_tr = dataset[dataset.indeks.isin(_X_tr.values)][y_col_name]
    X_te = X[dataset[dataset.indeks.isin(_X_te.values)].index]
    y_te = dataset[dataset.indeks.isin(_X_te.values)][y_col_name]
    return X_tr, X_te, y_tr, y_te


def train_model(model, X_tr, X_te, y_tr, y_te, batch_size=256):
    print(X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_tr, y_tr, batch_size=batch_size, epochs=200,
                        shuffle=True, validation_data=(X_te, y_te),
                        callbacks=[reduce_lr])
    return (model, history)


def train_model_adadelta(model, X_tr, X_te, y_tr, y_te):
    print(X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)
    model.compile(optimizer=adadelta, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_tr, y_tr, batch_size=256, epochs=200,
                        shuffle=True, validation_data=(X_te, y_te))
    return model, history


def train_model_SGD(model, X_tr, X_te, y_tr, y_te):
    print(X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_tr, y_tr, batch_size=256, epochs=200,
                        shuffle=True, validation_data=(X_te, y_te),
                        callbacks=[reduce_lr])
    return (model, history)


def ExperimentNConvLrs(n_iterations, dataset, logging_file):
    if not os.path.isfile(logging_file):
        create_logging_file(logging_file)

    X_tr, X_te, y_tr, y_te = preprocess_data(dataset, 'ekstrakcja', 'klasa_id', 'indeks')

    model_conv1 = \
        architecture_nconv.architecture_CONV_k3_f64_FC_basic(X_tr,
                                                             np.unique(y_tr).shape[0],
                                                             nb_conv=1, nb_fc=1)
    model_conv2 = \
        architecture_nconv.architecture_CONV_k3_f64_FC_basic(X_tr,
                                                             np.unique(y_tr).shape[0],
                                                             nb_conv=2, nb_fc=1)
    model_conv3 = \
        architecture_nconv.architecture_CONV_k3_f64_FC_basic(X_tr,
                                                             np.unique(y_tr).shape[0],
                                                             nb_conv=3, nb_fc=1)

    model_conv4 = \
        architecture_nconv.architecture_CONV_k9_f64_FC_basic(X_tr,
                                                             np.unique(y_tr).shape[0],
                                                             nb_conv=1, nb_fc=1)
    model_conv5 = \
        architecture_nconv.architecture_CONV_k9_f64_FC_basic(X_tr,
                                                             np.unique(y_tr).shape[0],
                                                             nb_conv=2, nb_fc=1)
    model_conv6 = \
        architecture_nconv.architecture_CONV_k9_f64_FC_basic(X_tr,
                                                             np.unique(y_tr).shape[0],
                                                             nb_conv=3, nb_fc=1)
    model_conv7 = \
        architecture_nconv.architecture_CONV_k12_f64_FC_basic(X_tr,
                                                              np.unique(y_tr).shape[0],
                                                              nb_conv=1, nb_fc=1)
    model_conv8 = \
        architecture_nconv.architecture_CONV_k12_f64_FC_basic(X_tr,
                                                              np.unique(y_tr).shape[0],
                                                              nb_conv=2, nb_fc=1)
    model_conv9 = \
        architecture_nconv.architecture_CONV_k12_f64_FC_basic(X_tr,
                                                              np.unique(y_tr).shape[0],
                                                              nb_conv=3, nb_fc=1)

    models_to_train = [model_conv1, model_conv2, model_conv3, model_conv4, model_conv5,
                       model_conv6, model_conv7, model_conv8, model_conv9]

    for nb, model in enumerate(models_to_train):
        for _iter in n_iterations:
            model_1, model_hist = train_model(model, X_tr, X_te, y_tr, y_te)

            y_pred = model_1.predict_classes(X_te)
            report = classification_report(y_te, y_pred, output_dict=True)
            print(report)

            with open(logging_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_1.name,
                                 report['4']['f1-score'], report['4']['precision'],
                                 report['4']['recall'], report['5']['f1-score'],
                                 report['5']['precision'], report['5']['recall'],
                                 report['6']['f1-score'],
                                 report['6']['precision'], report['6']['recall'],
                                 model_hist.history.get('acc')[-1], model_hist.history.get('val_acc')[-1],
                                 model_hist.history.get('loss')[-1], model_hist.history.get('val_loss')[-1]])

    return


def ExperimentNPoolLrs(n_iterations, dataset, logging_file, data_type, save=True):
    if not os.path.isfile(logging_file):
        print('logging file created')
        create_logging_file(logging_file)

    X_tr, X_te, y_tr, y_te = preprocess_data(dataset, 'ekstrakcja', 'klasa_id', 'indeks')

    model_pool1 = \
        architecture_pool.architecture_CONV_k3_f64_FC_pool(X_tr,
                                                           np.unique(y_tr).shape[0],
                                                           nb_conv=1, nb_fc=1)
    model_pool2 = \
        architecture_pool.architecture_CONV_k3_f32_FC_pool(X_tr,
                                                           np.unique(y_tr).shape[0],
                                                           nb_conv=1, nb_fc=2)
    model_pool3 = \
        architecture_pool.architecture_CONV_k3_f32_FC_pool(X_tr,
                                                           np.unique(y_tr).shape[0],
                                                           nb_conv=2, nb_fc=1)
    model_pool4 = \
        architecture_pool.architecture_CONV_k3_f32_FC_pool(X_tr,
                                                           np.unique(y_tr).shape[0],
                                                           nb_conv=2, nb_fc=2)
    model_pool5 = \
        architecture_pool.architecture_CONV_k9_f64_FC_pool(X_tr,
                                                           np.unique(y_tr).shape[0],
                                                           nb_conv=1, nb_fc=1)
    model_pool6 = \
        architecture_pool.architecture_CONV_k9_f64_FC_pool(X_tr,
                                                           np.unique(y_tr).shape[0],
                                                           nb_conv=1, nb_fc=2)
    model_pool7 = \
        architecture_pool.architecture_CONV_k9_f64_FC_pool(X_tr,
                                                           np.unique(y_tr).shape[0],
                                                           nb_conv=2, nb_fc=1)
    model_pool8 = \
        architecture_pool.architecture_CONV_k9_f64_FC_pool(X_tr,
                                                           np.unique(y_tr).shape[0],
                                                           nb_conv=2, nb_fc=2)
    model_pool9 = \
        architecture_pool.architecture_CONV_k12_f64_FC_pool(X_tr,
                                                            np.unique(y_tr).shape[0],
                                                            nb_conv=1, nb_fc=1)
    model_pool10 = \
        architecture_pool.architecture_CONV_k12_f64_FC_pool(X_tr,
                                                            np.unique(y_tr).shape[0],
                                                            nb_conv=1, nb_fc=2)
    model_pool11 = \
        architecture_pool.architecture_CONV_k12_f64_FC_pool(X_tr,
                                                            np.unique(y_tr).shape[0],
                                                            nb_conv=2, nb_fc=1)
    model_pool12 = \
        architecture_pool.architecture_CONV_k12_f64_FC_pool(X_tr,
                                                            np.unique(y_tr).shape[0],
                                                            nb_conv=2, nb_fc=2)

    models_to_train = [model_pool1, model_pool2, model_pool3, model_pool4, model_pool5,
                       model_pool6, model_pool7, model_pool8, model_pool9, model_pool10,
                       model_pool11, model_pool12]

    for nb, model in enumerate(models_to_train):
        for _iter in n_iterations:
            model_1, model_hist = train_model(model, X_tr, X_te, y_tr, y_te)

            y_pred = model_1.predict_classes(X_te)
            report = classification_report(y_te, y_pred, output_dict=True)
            print(report)

            with open(logging_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_1.name,
                                 report['4']['f1-score'], report['4']['precision'],
                                 report['4']['recall'], report['5']['f1-score'],
                                 report['5']['precision'], report['5']['recall'],
                                 report['6']['f1-score'],
                                 report['6']['precision'], report['6']['recall'],
                                 model_hist.history.get('acc')[-1], model_hist.history.get('val_acc')[-1],
                                 model_hist.history.get('loss')[-1], model_hist.history.get('val_loss')[-1]])

    return


def ExperimentRegularization(n_iterations, dataset, logging_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not os.path.isfile(logging_file):
        create_logging_file(logging_file)

    X_tr, X_te, y_tr, y_te = preprocess_data(dataset, 'ekstrakcja', 'klasa_id', 'indeks')

    # model with dropout set for 0.3
    model_conv1 = \
        architecture_regularization.architecture_CONV_FC_dropout(X_tr,
                                                                 np.unique(y_tr).shape[0],
                                                                 nb_conv=2, nb_fc=1,
                                                                 dropout_rate=0.3)
    # model with dropout set for 0.5
    model_conv2 = \
        architecture_regularization.architecture_CONV_FC_dropout(X_tr,
                                                                 np.unique(y_tr).shape[0],
                                                                 nb_conv=2, nb_fc=1,
                                                                 dropout_rate=0.5)
    # model with dropout set for 0.8
    model_conv3 = \
        architecture_regularization.architecture_CONV_FC_dropout(X_tr,
                                                                 np.unique(y_tr).shape[0],
                                                                 nb_conv=2, nb_fc=1,
                                                                 dropout_rate=0.8)
    # dropout 0.5 with l1 l2 regularization
    model_conv4 = \
        architecture_regularization.architecture_CONV_FC_dropout_L1_L2(X_tr,
                                                                       np.unique(y_tr).shape[0],
                                                                       nb_conv=2, nb_fc=1)
    # dropout 0.5 with l1 regularization
    model_conv5 = \
        architecture_regularization.architecture_CONV_FC_dropout_L1(X_tr,
                                                                    np.unique(y_tr).shape[0],
                                                                    nb_conv=2, nb_fc=1)
    # dropout 0.5 with l2 regularization
    model_conv6 = \
        architecture_regularization.architecture_CONV_FC_dropout_L2(X_tr,
                                                                    np.unique(y_tr).shape[0],
                                                                    nb_conv=2, nb_fc=1)
    # model with batch norm
    model_conv7 = \
        architecture_regularization.architecture_CONV_FC_batch_norm(X_tr,
                                                                    np.unique(y_tr).shape[0],
                                                                    nb_conv=2, nb_fc=1)
    # model with batch norm and elastic search regularization
    model_conv8 = \
        architecture_regularization.architecture_CONV_FC_batch_norm_L1_L2(X_tr,
                                                                          np.unique(y_tr).shape[0],
                                                                          nb_conv=2, nb_fc=1)
    # model with batch norm and l1 regularization
    model_conv9 = \
        architecture_regularization.architecture_CONV_FC_batch_norm_L1(X_tr,
                                                                       np.unique(y_tr).shape[0],
                                                                       nb_conv=2, nb_fc=1)

    # model with batch norm and l2 regularization
    model_conv11 = \
        architecture_regularization.architecture_CONV_FC_batch_norm_L2(X_tr,
                                                                       np.unique(y_tr).shape[0],
                                                                       nb_conv=2, nb_fc=1)

    # model with batch norm, dropout
    model_conv12 = \
        architecture_regularization.architecture_CONV_FC_batch_norm_dropout(X_tr,
                                                                            np.unique(y_tr).shape[0],
                                                                            nb_conv=2, nb_fc=1)
    # model with batch norm, dropout + elastic search
    model_conv13 = \
        architecture_regularization.architecture_CONV_FC_batch_norm_dropout_L1_l2(X_tr,
                                                                                  np.unique(y_tr).shape[0],
                                                                                  nb_conv=2, nb_fc=1)

    # model with batch norm, dropout + L1
    model_conv14 = \
        architecture_regularization.architecture_CONV_FC_batch_norm_dropout_L1(X_tr,
                                                                               np.unique(y_tr).shape[0],
                                                                               nb_conv=2, nb_fc=1)

    # model with batch norm, dropout + L2
    model_conv15 = \
        architecture_regularization.architecture_CONV_FC_batch_norm_dropout_L1(X_tr,
                                                                               np.unique(y_tr).shape[0],
                                                                               nb_conv=2, nb_fc=1)

    models_to_train = [model_conv1, model_conv2, model_conv3, model_conv4, model_conv5,
                       model_conv6, model_conv7, model_conv8, model_conv9,
                       model_conv11, model_conv12, model_conv13, model_conv14, model_conv15]

    for _iter in range(n_iterations):
        for nb, model in enumerate(models_to_train):
            print('training model {0} out {1}'.format(nb + 1, len(models_to_train)))
            model_1, model_hist = train_model(model, X_tr, X_te, y_tr, y_te)

            y_pred = model_1.predict_classes(X_te)
            report = classification_report(y_te, y_pred, output_dict=True)
            print(report)

            with open(logging_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_1.name,
                                 report['4']['f1-score'], report['4']['precision'],
                                 report['4']['recall'], report['5']['f1-score'],
                                 report['5']['precision'], report['5']['recall'],
                                 report['6']['f1-score'],
                                 report['6']['precision'], report['6']['recall'],
                                 model_hist.history.get('acc')[-1], model_hist.history.get('val_acc')[-1],
                                 model_hist.history.get('loss')[-1], model_hist.history.get('val_loss')[-1]])

    return


def ExperimentActivation(n_iterations, dataset, logging_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if not os.path.isfile(logging_file):
        create_logging_file(logging_file)

    X_tr, X_te, y_tr, y_te = preprocess_data(dataset, 'ekstrakcja', 'klasa_id', 'indeks')

    # model with dropout set for 0.3
    model_conv1 = \
        architecture_activation_and_initializers.architecture_CONV_FC_batch_norm_dropout_L1_l2(X_tr,
                                                                                               np.unique(y_tr).shape[0],
                                                                                               nb_conv=2, nb_fc=1)
    # model with dropout set for 0.5
    model_conv2 = \
        architecture_activation_and_initializers.architecture_CONV_FC_batch_norm_dropout_L1_l2_TANH(X_tr,
                                                                                                    np.unique(y_tr).shape[0],
                                                                                                    nb_conv=2, nb_fc=1)
    # model with dropout set for 0.8
    model_conv3 = \
        architecture_activation_and_initializers.architecture_CONV_FC_batch_norm_dropout_L1_l2_SIGMOID(X_tr,
                                                                                                       np.unique(y_tr).shape[0],
                                                                                                       nb_conv=2, nb_fc=1)
    # dropout 0.5 with l1 l2 regularization
    model_conv4 = \
        architecture_activation_and_initializers.architecture_CONV_FC_batch_norm_dropout_L1_l2_LEAKY_ReLU(X_tr,
                                                                                                          np.unique(y_tr).shape[0],
                                                                                                          nb_conv=2, nb_fc=1)
    # dropout 0.5 with l1 regularization
    model_conv5 = \
        architecture_activation_and_initializers.architecture_CONV_FC_batch_norm_dropout_L1_l2(X_tr,
                                                                                               np.unique(y_tr).shape[0],
                                                                                               nb_conv=2, nb_fc=1)
    # dropout 0.5 with l2 regularization
    model_conv6 = \
        architecture_activation_and_initializers.architecture_CONV_FC_batch_norm_dropout_L1_l2_XAVIER(X_tr,
                                                                                                      np.unique(y_tr).shape[0],
                                                                                                      nb_conv=2, nb_fc=1)
    # model with batch norm
    model_conv7 = \
        architecture_activation_and_initializers.architecture_CONV_FC_batch_norm_dropout_L1_l2_V_SCALLING(X_tr,
                                                                                                          np.unique(y_tr).shape[0],
                                                                                                          nb_conv=2, nb_fc=1)
    # model with batch norm and elastic search regularization
    model_conv8 = \
        architecture_activation_and_initializers.architecture_CONV_FC_batch_norm_dropout_L1_l2_H_NORMAL(X_tr,
                                                                                                        np.unique(y_tr).shape[0],
                                                                                                        nb_conv=2, nb_fc=1)
    # model with batch norm and l1 regularization
    model_conv9 = \
        architecture_activation_and_initializers.architecture_CONV_FC_batch_norm_dropout_L1_l2_H_UNIFORM(X_tr,
                                                                                                         np.unique(y_tr).shape[0],
                                                                                                         nb_conv=2, nb_fc=1)

    models_to_train = [model_conv1, model_conv2, model_conv3, model_conv4, model_conv5,
                        model_conv6, model_conv7, model_conv8, model_conv9]

    for nb, model in enumerate(models_to_train):
        for _iter in range(n_iterations):
            print('training model {0} out {1}'.format(nb + 1, len(models_to_train)))
            model_1, model_hist = train_model(model, X_tr, X_te, y_tr, y_te)

            y_pred = model_1.predict_classes(X_te)
            report = classification_report(y_te, y_pred, output_dict=True)
            print(report)

            with open(logging_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_1.name,
                                 report['4']['f1-score'], report['4']['precision'],
                                 report['4']['recall'], report['5']['f1-score'],
                                 report['5']['precision'], report['5']['recall'],
                                 report['6']['f1-score'],
                                 report['6']['precision'], report['6']['recall'],
                                 model_hist.history.get('acc')[-1], model_hist.history.get('val_acc')[-1],
                                 model_hist.history.get('loss')[-1], model_hist.history.get('val_loss')[-1]])

    return


def ExperimentOptimization(n_iterations, dataset, logging_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if not os.path.isfile(logging_file):
        create_logging_file(logging_file)

    X_tr, X_te, y_tr, y_te = preprocess_data(dataset, 'ekstrakcja', 'klasa_id', 'indeks')

    # model with batch norm and l1 regularization
    model_conv1 = \
        architecture_activation_and_initializers.architecture_CONV_FC_batch_norm_dropout_L1_l2_XAVIER(X_tr,
                                                                                                      np.unique(y_tr).shape[0],
                                                                                                      nb_conv=2, nb_fc=1)

    models_to_train = [model_conv1]

    for nb, model in enumerate(models_to_train):
        for _iter in range(n_iterations):
            print('training model {0} out {1}'.format(nb + 1, len(models_to_train)))
            model_1, model_hist = train_model(model, X_tr, X_te, y_tr, y_te)

            y_pred = model_1.predict_classes(X_te)
            report = classification_report(y_te, y_pred, output_dict=True)
            print(report)

            with open(logging_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_1.name,
                                 report['4']['f1-score'], report['4']['precision'],
                                 report['4']['recall'], report['5']['f1-score'],
                                 report['5']['precision'], report['5']['recall'],
                                 report['6']['f1-score'],
                                 report['6']['precision'], report['6']['recall'],
                                 model_hist.history.get('acc')[-1], model_hist.history.get('val_acc')[-1],
                                 model_hist.history.get('loss')[-1], model_hist.history.get('val_loss')[-1]])
            model_1 = None
            model_hist = None

            model_1, model_hist = train_model_adadelta(model, X_tr, X_te, y_tr, y_te)

            y_pred = model_1.predict_classes(X_te)
            report = classification_report(y_te, y_pred, output_dict=True)
            print(report)

            with open(logging_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_1.name + '_ADADELTA',
                                 report['4']['f1-score'], report['4']['precision'],
                                 report['4']['recall'], report['5']['f1-score'],
                                 report['5']['precision'], report['5']['recall'],
                                 report['6']['f1-score'],
                                 report['6']['precision'], report['6']['recall'],
                                 model_hist.history.get('acc')[-1], model_hist.history.get('val_acc')[-1],
                                 model_hist.history.get('loss')[-1], model_hist.history.get('val_loss')[-1]])
            model_1 = None
            model_hist = None

            model_1, model_hist = train_model_SGD(model, X_tr, X_te, y_tr, y_te)

            y_pred = model_1.predict_classes(X_te)
            report = classification_report(y_te, y_pred, output_dict=True)
            print(report)

            with open(logging_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_1.name + '_SGD',
                                 report['4']['f1-score'], report['4']['precision'],
                                 report['4']['recall'], report['5']['f1-score'],
                                 report['5']['precision'], report['5']['recall'],
                                 report['6']['f1-score'],
                                 report['6']['precision'], report['6']['recall'],
                                 model_hist.history.get('acc')[-1], model_hist.history.get('val_acc')[-1],
                                 model_hist.history.get('loss')[-1], model_hist.history.get('val_loss')[-1]])
            model_1 = None
            model_hist = None

    return


def ExperimentBatchSize(n_iterations, dataset, logging_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    batch_sizes = [10, 20, 50, 100, 200, 500]
    if not os.path.isfile(logging_file):
        create_logging_file(logging_file)

    X_tr, X_te, y_tr, y_te = preprocess_data(dataset, 'ekstrakcja', 'klasa_id', 'indeks')

    # model with batch norm and l1 regularization
    model_conv1 = \
        architecture_activation_and_initializers.architecture_CONV_FC_batch_norm_dropout_L1_l2_XAVIER(X_tr,
                                                                                                      np.unique(y_tr).shape[0],
                                                                                                      nb_conv=2, nb_fc=1)
    models_to_train = [model_conv1]

    for nb, model in enumerate(models_to_train):
        for _iter in range(n_iterations):
            for batch_size in batch_sizes:
                print('training model {0} out {1}'.format(nb + 1, len(models_to_train)))
                model_1, model_hist = train_model(model, X_tr, X_te, y_tr, y_te, batch_size=batch_size)
                y_pred = model_1.predict_classes(X_te)
                report = classification_report(y_te, y_pred, output_dict=True)
                print(report)

                with open(logging_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([model_1.name + '_' + str(_iter) + 'b_size' + str(batch_size),
                                     report['4']['f1-score'], report['4']['precision'],
                                     report['4']['recall'], report['5']['f1-score'],
                                     report['5']['precision'], report['5']['recall'],
                                     report['6']['f1-score'],
                                     report['6']['precision'], report['6']['recall'],
                                     model_hist.history.get('acc')[-1], model_hist.history.get('val_acc')[-1],
                                     model_hist.history.get('loss')[-1], model_hist.history.get('val_loss')[-1]])
                model_1 = None
                model_hist = None
    return
