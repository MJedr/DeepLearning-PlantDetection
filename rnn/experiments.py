import csv
import datetime
import os

import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import rnn.architectures as architectures
from cnn_callbacks import reduceLR


reduce_lr = reduceLR()
adam = Adam(lr=0.01, clipnorm=1.0)
rms_prop = RMSprop(rho=0.9)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


def reduceLR(patience=5, min_lr=0.000001):
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.6, patience=2, min_lr=min_lr
    )
    return reduce_lr


def create_logging_file(name=r"results/experiment_results.csv"):
    if os.path.isdir(r"results"):
        os.mkdir(r"results")
    fields = [
        "model_name",
        "F_u_f1",
        "F_u_PA",
        "F_u_UA",
        "M_c_f1",
        "M_c_PA",
        "M_c_UA",
        "Other_f1",
        "Other_PA",
        "Other_UA",
        "OA",
        "OA_WER",
        "LOSS",
        "LOSS_WER",
    ]

    with open(name, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        csvfile.close()

    return


def preprocess_data(dataset, x_col_name, y_col_name, polygon_ind_col):
    X = dataset[x_col_name].apply(lambda x: np.array(x)).values
    X = np.concatenate(X).reshape(X.shape[0], X[1].shape[0], 1)
    X = (X) / (np.max(X))
    # split dataset according to polygon numbers -
    # we don't want samples from same polygon in tain and test set
    unique_indexes = dataset.drop_duplicates(polygon_ind_col)
    _X_tr, _X_te, y_tr, y_te = train_test_split(
        unique_indexes[polygon_ind_col], unique_indexes[y_col_name]
    )
    X_tr = X[dataset[dataset.indeks.isin(_X_tr.values)].index]
    y_tr = dataset[dataset.indeks.isin(_X_tr.values)][y_col_name].values
    X_te = X[dataset[dataset.indeks.isin(_X_te.values)].index]
    y_te = dataset[dataset.indeks.isin(_X_te.values)][y_col_name].values
    return X_tr, X_te, y_tr, y_te


def train_model(model, X_tr, X_te, y_tr, y_te):
    print(X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)
    model.compile(
        optimizer=adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        X_tr,
        y_tr,
        batch_size=256,
        epochs=100,
        shuffle=True,
        validation_data=(X_te, y_te),
        callbacks=[reduce_lr],
    )
    print(history.history["acc"])
    print(history.history["val_acc"])
    print(history.history["loss"])
    print(history.history["val_loss"])
    return (model, history)


def train_model_rmsprop(model, X_tr, X_te, y_tr, y_te):
    print(X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)
    model.compile(
        optimizer=rms_prop, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        X_tr,
        y_tr,
        batch_size=256,
        epochs=100,
        shuffle=True,
        validation_data=(X_te, y_te),
        callbacks=[reduce_lr],
    )
    print(history.history["acc"])
    print(history.history["val_acc"])
    print(history.history["loss"])
    print(history.history["val_loss"])
    return (model, history)


def train_model_sgd(model, X_tr, X_te, y_tr, y_te):
    print(X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)
    model.compile(
        optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        X_tr,
        y_tr,
        batch_size=256,
        epochs=100,
        shuffle=True,
        validation_data=(X_te, y_te),
        callbacks=[reduce_lr],
    )
    print(history.history["acc"])
    print(history.history["val_acc"])
    print(history.history["loss"])
    print(history.history["val_loss"])
    return (model, history)


def ExperimentArchitecture(n_iterations, dataset, logging_file):
    if not os.path.isfile(logging_file):
        create_logging_file(logging_file)

    X_tr, X_te, y_tr, y_te = preprocess_data(
        dataset, "ekstrakcja", "klasa_id", "indeks"
    )
    print(X_tr, y_tr)

    model_rnn1 = architectures.architecture_rnn_dropout(X_tr, 1, 128, 1, 128)

    model_rnn2 = architectures.architecture_rnn_dropout(X_tr, 1, 16, 1, 128)

    model_rnn3 = architectures.architecture_rnn_dropout(X_tr, 2, 16, 1, 128)

    model_rnn4 = architectures.architecture_rnn_dropout(X_tr, 1, 16, 2, 128)

    model_rnn5 = architectures.architecture_rnn_dropout(X_tr, 2, 16, 2, 128)

    model_rnn6 = architectures.architecture_rnn_dropout(X_tr, 1, 16, 1, 256)

    model_rnn7 = architectures.architecture_rnn_dropout(X_tr, 2, 16, 1, 256)

    model_rnn8 = architectures.architecture_rnn_dropout(X_tr, 1, 16, 2, 256)

    model_rnn9 = architectures.architecture_rnn_dropout(X_tr, 2, 16, 2, 256)

    model_rnn10 = architectures.architecture_rnn_cascade(X_tr, 2, 16, 1, 256)

    model_rnn11 = architectures.architecture_rnn_cascade(X_tr, 3, 16, 1, 256)

    model_rnn12 = architectures.architecture_rnn_cascade(X_tr, 3, 32, 1, 256)

    models_to_train = [
        model_rnn1,
        model_rnn2,
        model_rnn3,
        model_rnn4,
        model_rnn5,
        model_rnn6,
        model_rnn7,
        model_rnn8,
        model_rnn9,
        model_rnn10,
        model_rnn11,
        model_rnn12,
    ]

    for nb, model in enumerate(models_to_train):
        for _iter in range(n_iterations):
            print("training model {0} out {1}".format(nb + 1, len(models_to_train)))
            model_1, model_hist = train_model(model, X_tr, X_te, y_tr, y_te)
            y_pred = model_1.predict_classes(X_te)
            report = classification_report(y_te, y_pred, output_dict=True)
            print(report)

            with open(logging_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        model_1.name + "_" + str(_iter),
                        report["4"]["f1-score"],
                        report["4"]["precision"],
                        report["4"]["recall"],
                        report["5"]["f1-score"],
                        report["5"]["precision"],
                        report["5"]["recall"],
                        report["6"]["f1-score"],
                        report["6"]["precision"],
                        report["6"]["recall"],
                        model_hist.history.get("acc")[-1],
                        model_hist.history.get("val_acc")[-1],
                        model_hist.history.get("loss")[-1],
                        model_hist.history.get("val_loss")[-1],
                    ]
                )
            model_1 = None
            model_hist = None
    return


def ExperimentArchitecturesRegularizationActivation(
    n_iterations, dataset, logging_file
):
    if not os.path.isfile(logging_file):
        create_logging_file(logging_file)

    X_tr, X_te, y_tr, y_te = preprocess_data(
        dataset, "ekstrakcja", "klasa_id", "indeks"
    )
    print(X_tr, y_tr)

    # model with batch norm and l1 regularization

    model_rnn1 = architectures.architecture_rnn_batch_norm_L1_l2(X_tr, 1, 16, 1, 128)

    model_rnn2 = architectures.architecture_rnn_batch_norm_L1_l2(X_tr, 1, 16, 1, 256)

    model_rnn3 = architectures.architecture_rnn_batch_norm_L1_l2(X_tr, 2, 16, 1, 256)

    model_rnn4 = architectures.architecture_rnn_cascade_regularization_bn(
        X_tr, 2, 16, 1, 256
    )

    model_rnn5 = architectures.architecture_rnn_cascade_regularization_bn(
        X_tr, 3, 32, 1, 256
    )

    models_to_train = [model_rnn1, model_rnn2, model_rnn3, model_rnn4, model_rnn5]

    for nb, model in enumerate(models_to_train):
        for _iter in range(n_iterations):
            print("training model {0} out {1}".format(nb + 1, len(models_to_train)))
            model_1, model_hist = train_model(model, X_tr, X_te, y_tr, y_te)
            y_pred = model_1.predict_classes(X_te)
            report = classification_report(y_te, y_pred, output_dict=True)
            print(report)

            with open(logging_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        model_1.name + "_" + str(_iter),
                        report["4"]["f1-score"],
                        report["4"]["precision"],
                        report["4"]["recall"],
                        report["5"]["f1-score"],
                        report["5"]["precision"],
                        report["5"]["recall"],
                        report["6"]["f1-score"],
                        report["6"]["precision"],
                        report["6"]["recall"],
                        model_hist.history.get("acc")[-1],
                        model_hist.history.get("val_acc")[-1],
                        model_hist.history.get("loss")[-1],
                        model_hist.history.get("val_loss")[-1],
                    ]
                )
            model_1 = None
            model_hist = None

            model_1, model_hist = train_model_rmsprop(model, X_tr, X_te, y_tr, y_te)
            y_pred = model_1.predict_classes(X_te)
            report = classification_report(y_te, y_pred, output_dict=True)
            print(report)

            with open(logging_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        model_1.name + "_" + str(_iter) + "rms_prop",
                        report["4"]["f1-score"],
                        report["4"]["precision"],
                        report["4"]["recall"],
                        report["5"]["f1-score"],
                        report["5"]["precision"],
                        report["5"]["recall"],
                        report["6"]["f1-score"],
                        report["6"]["precision"],
                        report["6"]["recall"],
                        model_hist.history.get("acc")[-1],
                        model_hist.history.get("val_acc")[-1],
                        model_hist.history.get("loss")[-1],
                        model_hist.history.get("val_loss")[-1],
                    ]
                )
            model_1 = None
            model_hist = None

            model_1, model_hist = train_model_sgd(model, X_tr, X_te, y_tr, y_te)
            y_pred = model_1.predict_classes(X_te)
            report = classification_report(y_te, y_pred, output_dict=True)
            print(report)

            with open(logging_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        model_1.name + "_" + str(_iter) + "sdg",
                        report["4"]["f1-score"],
                        report["4"]["precision"],
                        report["4"]["recall"],
                        report["5"]["f1-score"],
                        report["5"]["precision"],
                        report["5"]["recall"],
                        report["6"]["f1-score"],
                        report["6"]["precision"],
                        report["6"]["recall"],
                        model_hist.history.get("acc")[-1],
                        model_hist.history.get("val_acc")[-1],
                        model_hist.history.get("loss")[-1],
                        model_hist.history.get("val_loss")[-1],
                    ]
                )
            model_1 = None
            model_hist = None
    return


def ExperimentArchitectureRecurrentDropout(n_iterations, dataset, logging_file):
    log_dir = "logs//" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if not os.path.isfile(logging_file):
        create_logging_file(logging_file)

    X_tr, X_te, y_tr, y_te = preprocess_data(
        dataset, "ekstrakcja", "klasa_id", "indeks"
    )
    print(X_tr, y_tr)

    # model with batch norm and l1 regularization
    model_rnn = architectures.architecture_rnn_cascade_regularization_bn_dropout(
        X_tr, 2, 16, 1, 256
    )

    model_rnn1 = architectures.architecture_rnn_cascade_regularization_bn_dropout(
        X_tr, 2, 16, 1, 256, dr=0.2
    )

    models_to_train = [model_rnn, model_rnn1]
    for nb, model in enumerate(models_to_train):
        for _iter in range(n_iterations):
            print("training model {0} out {1}".format(nb + 1, len(models_to_train)))
            model.compile(
                optimizer=adam,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            model_hist = model.fit(
                X_tr,
                y_tr,
                batch_size=256,
                epochs=100,
                shuffle=True,
                validation_data=(X_te, y_te),
                callbacks=[reduce_lr, tensorboard_callback],
            )
            y_pred = model.predict_classes(X_te)
            report = classification_report(y_te, y_pred, output_dict=True)
            print(report)

            with open(logging_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        model.name + "_" + str(_iter),
                        report["4"]["f1-score"],
                        report["4"]["precision"],
                        report["4"]["recall"],
                        report["5"]["f1-score"],
                        report["5"]["precision"],
                        report["5"]["recall"],
                        report["6"]["f1-score"],
                        report["6"]["precision"],
                        report["6"]["recall"],
                        model_hist.history.get("acc")[-1],
                        model_hist.history.get("val_acc")[-1],
                        model_hist.history.get("loss")[-1],
                        model_hist.history.get("val_loss")[-1],
                    ]
                )
            model_1 = None
            model_hist = None
    return


def ExperimentArchitectureRecurrentRegularization(n_iterations, dataset, logging_file):
    log_dir = "logs//" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    if not os.path.isfile(logging_file):
        create_logging_file(logging_file)

    X_tr, X_te, y_tr, y_te = preprocess_data(
        dataset, "ekstrakcja", "klasa_id", "indeks"
    )
    print(X_tr, y_tr)

    # model with batch norm and l1 regularization
    model_rnn = architectures.architecture_rnn_cascade_recurrent_regularization_bn_(
        X_tr, 2, 16, 1, 256
    )

    model_rnn1 = architectures.architecture_rnn_cascade_recurrent_regularization_bn_(
        X_tr, 2, 16, 1, 256, dr=0.2
    )

    models_to_train = [model_rnn, model_rnn1]

    for nb, model in enumerate(models_to_train):
        for _iter in range(n_iterations):
            print("training model {0} out {1}".format(nb + 1, len(models_to_train)))
            model.compile(
                optimizer=adam,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            model_hist = model.fit(
                X_tr,
                y_tr,
                batch_size=256,
                epochs=100,
                shuffle=True,
                validation_data=(X_te, y_te),
                callbacks=[reduce_lr, tensorboard_callback],
            )
            y_pred = model.predict_classes(X_te)
            report = classification_report(y_te, y_pred, output_dict=True)
            print(report)

            with open(logging_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        model.name + "_" + str(_iter),
                        report["4"]["f1-score"],
                        report["4"]["precision"],
                        report["4"]["recall"],
                        report["5"]["f1-score"],
                        report["5"]["precision"],
                        report["5"]["recall"],
                        report["6"]["f1-score"],
                        report["6"]["precision"],
                        report["6"]["recall"],
                        model_hist.history.get("acc")[-1],
                        model_hist.history.get("val_acc")[-1],
                        model_hist.history.get("loss")[-1],
                        model_hist.history.get("val_loss")[-1],
                    ]
                )
            model_1 = None
            model_hist = None
    return
