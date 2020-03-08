import psutil

from keras.models import model_from_json
import numpy as np
import pickle
import gdal, gdalnumeric
from gdalconst import *
import matplotlib.pyplot as plt
import os
from keras.models import model_from_json
from keras.optimizers import Adam
import tensorflow as tf



def get_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info()

print('zaladowany')
def classify_raster(raster_path, model_path, model_weights, out_raster_name, x_block_size=256, y_block_size=160):
    with tf.device('/gpu:0'):
        ds = gdal.Open(raster_path, GA_ReadOnly)
        # load json and create model
        adam = Adam(beta_1=0.9, beta_2=0.999, amsgrad=False)
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(model_weights)
        model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print("Loaded model from disk")
        band = ds.GetRasterBand(1)
        b_array = band.ReadAsArray()
        x_im_size = band.XSize
        y_im_size = band.YSize
        x_block_size = int(x_block_size)
        y_block_size = int(y_block_size)
        xsize = b_array.shape[1]
        ysize = b_array.shape[0]
        xstride = np.floor(xsize / x_block_size).astype('int64')
        ystride = np.floor(ysize / y_block_size).astype('int64')
        print(xstride, ystride)
        b_array = None

        out_raster = np.zeros([ysize, xsize])

        memory_driver = gdal.GetDriverByName('GTiff')
        proj = ds.GetProjectionRef()
        ext = ds.GetGeoTransform()
        out_raster_ds = memory_driver.Create(out_raster_name, x_im_size, y_im_size, 1, gdal.GDT_UInt16)
        out_raster_ds.SetProjection(proj)
        out_raster_ds.SetGeoTransform(ext)

        xsize = ds.GetRasterBand(1).XSize
        ysize = ds.GetRasterBand(1).YSize
        pixels = 0
        for xx in range(0, xstride):
            for yy in range(0, ystride):
                y = xx * x_block_size
                x = yy * y_block_size
                array = ds.ReadAsArray(y, x, y_block_size, x_block_size)
                canals = array.shape[0]
                temp = []
                for m in range(array.shape[1]):
                    for n in range(array.shape[2]):
                        temp.append(array[:, m, n].reshape(canals, 1))
                temp = np.array(temp)
                pred = model.predict_classes(temp)
                pred = pred.reshape(x_block_size, y_block_size)
                out_raster[x:x + x_block_size, y:y + y_block_size] = pred
                print('classified {0} pixels out of {1}'.format(pixels, xsize * ysize))
                x += x_block_size
                print('-------------------------------------')
            y += y_block_size
            print(xx, yy)
            if xx == (xstride - 1):
                if xsize % x_block_size != 0 and ysize % y_block_size != 0:
                    print('opcja 1')
                    array = ds.ReadAsArray(0, int(y_block_size*ystride), xsize, int(ysize%y_block_size))
                    temp = []
                    for m in range(array.shape[1]):
                        for n in range(array.shape[2]):
                            temp.append(array[:, m, n].reshape(canals, 1))
                    temp = np.array(temp)
                    pred = model.predict_classes(temp)
                    print(pred.shape)
                    pred = pred.reshape(ysize%y_block_size, xsize)
                    out_raster[ystride*y_block_size: ysize,
                    0:xsize] = pred
                    new_x_stride = int(xsize - (xstride * x_block_size))
                    new_x = int(xstride * x_block_size)
                    new_y = 0
                    for yyy in range(0, ystride):
                        array = ds.ReadAsArray(new_x,new_y, new_x_stride, y_block_size)
                        temp = []
                        for m in range(array.shape[1]):
                            for n in range(array.shape[2]):
                                temp.append(array[:, m, n].reshape(canals, 1))
                        temp = np.array(temp)
                        pred = model.predict_classes(temp)
                        print(pred.shape)
                        pred = pred.reshape(y_block_size, xsize - new_x)
                        print(new_x)
                        out_raster[new_y:new_y + y_block_size, new_x:xsize] = pred
                        new_y += y_block_size
                elif xsize % x_block_size != 0 and ysize % y_block_size == 0:
                    print('opcja2')
                    array = ds.ReadAsArray(0, int(xstride*x_block_size), int(ysize), int(xsize % x_block_size))
                    temp = np.zeros([ysize, int(xsize % x_block_size)])
                    for n in range(array.shape[2]):
                        for m in range(array.shape[1]):
                            p = array[:, m, n].reshape(1, -1)
                            p = model.predict(p)
                            temp[m, n] = p
                    out_raster[0:ysize, x_block_size * xstride: x_block_size * xstride + xsize % x_block_size] = temp
                elif xsize % x_block_size == 0 and ysize % y_block_size != 0:
                    array = ds.ReadAsArray(y, 0, int(ysize % y_block_size), int(xsize))
                    temp = np.zeros([ysize, int(xsize % x_block_size)])
                    print(temp.shape, array.shape)
                    for n in range(array.shape[2]):
                        for m in range(array.shape[1]):
                            p = array[:, m, n].reshape(1, -1)
                            p = model.predict(p)
                            temp[m, n] = p
                    out_raster[y_block_size * ystride: y_block_size * ystride + ysize % y_block_size, 0:xsize] = temp
                print('classified {0} pixels out of {1}'.format(pixels, xsize * ysize))
                y += y_block_size

        outband = out_raster_ds.GetRasterBand(1)
        outband.WriteArray(out_raster)
        outband.FlushCache()
        try:
            plt.imshow(out_raster)
            plt.savefig(out_raster_name + 'tiff')
        except:
            pass
        

        ds = None
        out_raster = None
        ds = None

        return

imagek5 = r'D:\k5\MOZ\KR1_K5_HS_MOZ.dat'
image = r'D:\time_serie\KR1_HS_K456'
model = r'models/2CONV_1k3f64_FC256_dropout0.5batch_normpool5_ts.json'
weights =r'models/2CONV_1k3f64_FC256_dropout0.5batch_normpool5_ts.h5'

model_ts_man = r'models_image_clf/modelts_ms.json'
model_ts_man_weights = r'models_image_clf/modelts_ms.h5'
model_k5_man = r'models_image_clf/modelk5_ms.json'
model_k5_man_weights = r'models_image_clf/modelk5_ms.h5'
model_ts_rs =  r'models_image_clf/modelts_rs.json'
model_ts_rs_weights = r'models_image_clf/modelts_rs.h5'
model_k5_rs = r'models_image_clf/modelk5_rs.json'
model_k5_rs_weights =  r'models_image_clf/modelk5_rs.h5'

paths = [model_ts_man, model_ts_man_weights, model_k5_man, model_k5_man_weights,
        model_ts_rs, model_ts_rs_weights, model_k5_rs, model_k5_rs_weights]

for path in paths:
    assert(os.path.exists(path))

# ts manual search
# classify_raster(image, model_ts_man, model_ts_man_weights, 'classification/ts_manual_search', 250, 250)

# k5 manual search
# classify_raster(imagek5, model_k5_man, model_k5_man_weights, 'classification/k5_manual_search', 250, 250)

# ts random search
# classify_raster(image, model_ts_rs, model_ts_rs_weights, 'classification/ts_random_search', 250, 250)

# k5 random search
classify_raster(imagek5, model_k5_rs, model_k5_rs_weights, 'classification/k5_random_search', 250, 250)
