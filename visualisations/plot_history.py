import matplotlib.pyplot as plt
import os
import datetime


def plot_history(history, model_name, data_type):
    accuracy_plot_path_exists = os.path.isfile('plots/{0}_{1}accuracy.png'.format(model_name, data_type))
    if accuracy_plot_path_exists:
        prefix = str(datetime.datetime.now()).replace(":", "-").replace(" ", "-").replace(".", "-")
        accuracy_plot_path = 'plots/{0}_{1}accuracy_{2}.png'.format(model_name, data_type, prefix)
        loss_plot_path = 'plots/{0}_{1}loss_{2}.png'.format(model_name, data_type, prefix)
    else:
        accuracy_plot_path = 'plots/{0}_{1}accuracy.png'.format(model_name, data_type)
        loss_plot_path = 'plots/{0}_{1}accuracy.png'.format(model_name, data_type)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(accuracy_plot_path)
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(loss_plot_path)
    plt.clf()
    return
