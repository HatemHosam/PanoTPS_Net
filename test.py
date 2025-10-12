import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from spatial_transformer import ElasticTransformer
from tensorflow.keras.applications.xception import preprocess_input, Xception
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import *

#test paths for images, edge, and corner maps
test_path = 'PanoContext/test/images/'
test_edge_path = 'PanoContext/test/edges/'
test_corner_path = 'PanoContext/test/edges/'

# read test image
img = cv2.imread(test_path+'PCts_0000.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img1 =  np.array(img,copy=True)
img = img[tf.newaxis,...]
img = preprocess_input(img)

# edge_gt = cv2.imread(test_edges_path+'PCts_0000.png')
# edge_gt = cv2.cvtColor(edge_gt, cv2.COLOR_BGR2RGB)

# corner_gt = cv2.imread(test_corner_path+'PCts_0000.png')
 
#read reference edge map
ref_img = tf.io.read_file('ref_edge.png')
ref_img = tf.io.decode_png(ref_img)
ref_img = tf.cast(ref_img, tf.float32)
ref_img = ref_img[tf.newaxis,...]

#read reference corner map
ref_img2 = tf.io.read_file('ref_corner.png')
ref_img2 = tf.io.decode_png(ref_img2)
ref_img2 = tf.image.rgb_to_grayscale(ref_img2)
ref_img2 = tf.cast(ref_img2, tf.float32)
ref_img2 = ref_img2[tf.newaxis,...]


#build MXception model
base_model = Xception(include_top=False, weights="imagenet", input_shape= (512,1024,3), pooling = 'avg')
blk_13 = base_model.get_layer('block13_sepconv2_act').output
avg = GlobalAveragePooling2D()(blk_13)
theta = Dense(2*16)(avg)
#add Thin-plate spline (Elastic) transformation layer for edge map
stl = ElasticTransformer((512,1024,3)).transform(ref_img, theta)
#add Thin-plate spline (Elastic) transformation layer for corner map
st2= ElasticTransformer((512,1024,1)).transform(ref_img2, theta)
model = Model(base_model.input, [stl, st2])

model.summary()
# load weights of pretrained model
model.load_weights('weights/weights_PanoTPS-Net_PanoContext+Whole_Stanford2D3D.h5')

# predict edge and corner maps
edge, corner = model.predict(img)

# process maps to be  visualized
edge = np.array(edge[0,:,:,:], dtype=np.uint8)
corner = np.array(corner[0,:,:,0], dtype=np.uint8)

#visualize the predicted edge and corner maps
plt.figure('predicted_edge_map')
plt.imshow(out)
plt.figure('predicted_Corner_map')
plt.imshow(edge)
plt.figure('img')
plt.imshow(img1)
plt.show()

