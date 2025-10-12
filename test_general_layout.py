import os
from xml.etree import ElementTree
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from spatial_transformer import ElasticTransformer
from tensorflow.keras.applications.xception import preprocess_input, Xception
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import *
from scipy.interpolate import interp1d
import random
import scipy.io
import math
from scipy import ndimage
from sklearn.utils import shuffle
from scipy import ndimage
import json


def separate_corners(x_min_val, x_max_val, width_val, y_cen_val, corner_map, edge_map):
    final_corners_2_split = round(width_val/ 50.)
     
    upper_y_val = y_cen_val
    lower_y_val = 512 - y_cen_val
    
    # draw black line to split wide corner to multiple corners    
    for k in range(final_corners_2_split-1):
       cv2.line(corner_map, (x_min_val+((k+1)*50), 0), (x_min_val+((k+1)*50), 1023), 0, 5) 
       cv2.line(edge_map, (x_min_val+((k+1)*50), upper_y_val), (x_min_val+((k+1)*50), lower_y_val), (0,0,0), 5) 
     
    return corner_map, edge_map
     
def round_to_nearest_base(x, base=50):
    return base * round(x/base)     

def corner_postproc(corner_map, edge_map):
    
    processed_corner_map = np.array(corner_map, copy=True)
    processed_edge_map = np.array(edge_map, copy=True)
    
    binary_corner_map = np.array(processed_corner_map, copy=True)
    binary_corner_map[binary_corner_map>=50] = 255
    binary_corner_map[binary_corner_map<50] = 0
    
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)  # this defines the connection filter
    labels, nb = ndimage.label(binary_corner_map, structure)
    width = 0
    for i in range(int(nb/2)):
       corner_temp = np.where(labels == i+1)
       x_max = np.amax(corner_temp[1])
       x_min = np.amin(corner_temp[1])
       # y _center for separation purpose
       y_max = np.amax(corner_temp[0])
       y_min = np.amin(corner_temp[0])
       
       y_cen = int((y_max+y_min)/2)
   
       
       width = round_to_nearest_base(x_max - x_min)
       if width > 50:
           processed_cor_map, processed_edge_map = separate_corners(x_min, x_max, width, y_cen, processed_corner_map, processed_edge_map)
    
    return processed_cor_map, processed_edge_map, labels


val_path = 'path/to/image_folder/'
val_edges_path = 'path/to//edge_maps/'
val_corners_path = 'path/to/corner_maps/'

img = cv2.imread(val_path+'7y3sRwLe3Va_ad77d6eeca5b492b8fe3317177f4f03f.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img1 =  np.array(img,copy=True)
img = img[tf.newaxis,...]
img = preprocess_input(img)

edge = cv2.imread(val_edges_path+'7y3sRwLe3Va_ad77d6eeca5b492b8fe3317177f4f03f.png')
edge = cv2.cvtColor(edge, cv2.COLOR_BGR2RGB)

corner = cv2.imread(val_corners_path+'7y3sRwLe3Va_ad77d6eeca5b492b8fe3317177f4f03f.png',0)
 
ref_img = tf.io.read_file('PC_0016_ref_edge.png')
ref_img = tf.io.decode_png(ref_img)[:,:,:3]
ref_img = tf.cast(ref_img, tf.float32)
ref_img = ref_img[tf.newaxis,...]
print(ref_img.shape)

ref_img2 = tf.io.read_file('PC_0016_ref_corner.png')
ref_img2 = tf.io.decode_png(ref_img2)[:,:,:3]
ref_img2 = tf.image.rgb_to_grayscale(ref_img2)
ref_img2 = tf.cast(ref_img2, tf.float32)
ref_img2 = ref_img2[tf.newaxis,...]
print(ref_img2.shape)


base_model = Xception(include_top=False, weights="imagenet", input_shape= (512,1024,3), pooling = 'avg')
blk_13 = base_model.get_layer('block13_sepconv2_act').output
avg = GlobalAveragePooling2D()(blk_13)
theta = Dense(2*64)(avg)
stl = ElasticTransformer((512,1024,3)).transform(ref_img, theta)

st2= ElasticTransformer((512,1024,1)).transform(ref_img2, theta)
model = Model(base_model.input, [stl, st2])

# model.summary()
model.load_weights('weights.h5')

edges, corners= model.predict(img)

edges = np.array(edges[0,:,:,:], dtype=np.uint8)
corners = np.array(corners[0,:,:,0], dtype=np.uint8)

processed_corners, processed_edges, labelled_img = corner_postproc(corners, edges)

plt.imsave('results/gt_edges.png',edge)
plt.imsave('results/pred_edges.png',edges)
plt.imsave('results/gt_corners.png',corner)
plt.imsave('results/pred_corners.png',corners)
plt.imsave('results/processed_corner_map.png',processed_corners)
plt.imsave('results/processed_edge_map.png',processed_edges)
plt.imsave('results/labeled_img.png',labelled_img)
plt.imsave('results/img.png',img1)



