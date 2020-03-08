from keras.callbacks import ReduceLROnPlateau, EarlyStopping

def reduceLR(patience=5, min_lr=0.000001):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6,
                                  patience=2, min_lr=min_lr)
    return reduce_lr

def earlyStop(patience=2):
    es = EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=patience,
                       verbose=0, mode='auto')
    return es
