# -*- coding: utf-8 -*-
import rasterize_dataset
from config import Config

# reference data with training polygons
data_shp = Config.data_shp
# reference image to extract samples and classify
data_raster = Config.data_raster
# output mask file name
mask_raster = r'outputs\maskaKR1k5.tiff'
# filenames for extracted values

export_csv = r'outputs\extracted_k5.csv'
export_pickle = r'outputs\extracted_przyklad_k5.pickle'

class_names = (rasterize_dataset.get_class_names(data_shp))
print('classes to be classified - ', class_names)
class_nb = len(class_names)
# copy of input shp file (to add unique indices for each polygon)
shp_copy = rasterize_dataset.create_index_fld(data_shp, class_names)
# rasterization
rasterize_dataset.rasterize(data_raster, mask_raster, shp_copy)
# training data extraction
rasterize_dataset.extract(data_raster, mask_raster, class_nb, class_names, export_pickle, export_csv)
