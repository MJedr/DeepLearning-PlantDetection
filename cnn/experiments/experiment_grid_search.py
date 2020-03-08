from cnn.architectures import architecture_pool, architecture_regularization1
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from cnn_callbacks import reduceLR, earlyStop
import csv
import os
import datetime
from sklearn.metrics import classification_report, confusion_matrix
from keras.optimizers import adam
from visualisations.plot_history import plot_history
import random
import json

# callbacks
reduce_lr = reduceLR()
early_stop = earlyStop()
adam = adam(lr=0.01, clipnorm=1.)

param_dict = {'conv_blocks': [2],
              'conv_filters': [16, 32, 64],
              'conv_kernel': [3, 9],
              'pool_kernel': [3, 9],
              'dropout': [0.2, 0.5, 0.7],
              'nb_fc': [1, 2],
              'nb_neurons': [128, 256, 512],
              'weight_initializer': ['he_normal', 'glorot_normal'],
              'l1_l2': [True, False]}


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


def train_model(model, X_tr, X_te, y_tr, y_te):
    print(X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_tr, y_tr, batch_size=256, epochs=200,
                        shuffle=True, validation_data=(X_te, y_te),
                        callbacks=[reduce_lr])
    print(history.history['acc'])
    print(history.history['val_acc'])
    print(history.history['loss'])
    print(history.history['val_loss'])
    return (model, history)


def ExperimentGridSearch(n_iterations, dataset, logging_file, data_type, save=True):
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    if not os.path.isfile(logging_file):
        print('logging file created')
        create_logging_file(logging_file)
    if not os.path.isdir(r'../results/architecture/'):
        os.mkdir(r'../results/architecture/')
    if not os.path.isdir(r'../models/'):
        os.mkdir(r'../results/models/')
    if not os.path.isdir(r"../results/confusion_matrix/"):
        os.mkdir(r"../results/confusion_matrix/")
    X_tr, X_te, y_tr, y_te = preprocess_data(dataset, 'ekstrakcja', 'klasa_id', 'indeks')

    for u in range(2):
        model_architecture = get_model_params(param_dict)
        print(model_architecture)
        model_name = 'model_' + str(u + 17) + data_type
        save_architecture(model_architecture, r'../results/architecture/' + model_name)
        for _iteration in range(n_iterations):
            model_name_iteration = model_name + str(_iteration)
            start_time = datetime.datetime.now()
            model = architecture_regularization1.architecture_basic(X_tr, np.unique(y_tr).shape[0],
                                                                    nb_conv=model_architecture['conv_blocks'],
                                                                    nb_fc=model_architecture['nb_fc'],
                                                                    dropout_rate=model_architecture['dropout'],
                                                                    conv_filters=model_architecture['conv_filters'],
                                                                    pool_kernel=model_architecture['pool_kernel'],
                                                                    conv_kernel=model_architecture['conv_kernel'],
                                                                    nb_neurons=model_architecture['nb_neurons'],
                                                                    weight_initializer=model_architecture[
                                                                       'weight_initializer'])

            model_1, model_hist = train_model(model, X_tr, X_te, y_tr, y_te)
            end_time = datetime.datetime.now()
            plot_history(model_hist, model_name_iteration, data_type=data_type)

            y_pred = model_1.predict_classes(X_te)
            report = classification_report(y_te, y_pred, output_dict=True)
            print(report)

            if save:
                cm = pd.DataFrame(confusion_matrix(y_te, y_pred))
                prefix = str(datetime.datetime.now()).replace(":", "-").replace(" ", "-").replace(".", "-")
                cm.to_csv(f""""..//results//confusion_matrix//cm_{model_name_iteration}_{prefix}.csv""")

                with open(logging_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([model_name_iteration,
                                     report['4']['f1-score'], report['4']['precision'],
                                     report['4']['recall'], report['5']['f1-score'],
                                     report['5']['precision'], report['5']['recall'],
                                     report['6']['f1-score'],
                                     report['6']['precision'], report['6']['recall'],
                                     model_hist.history.get('acc')[-1], model_hist.history.get('val_acc')[-1],
                                     model_hist.history.get('loss')[-1], model_hist.history.get('val_loss')[-1]])

                # serialize model to JSON
                model_json = model.to_json()
                with open(r"../models/{0}_{1}.json".format(model_name_iteration, data_type), "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                model_1.save_weights(r"../results/models/{0}_{1}.h5".format(model_name_iteration, data_type))
                print("Saved model to disk")
                model_1 = None
                model_hist = None
    return
