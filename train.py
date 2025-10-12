import os
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from spatial_transformer import ElasticTransformer
from tensorflow.keras.applications.xception import preprocess_input, Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import random
from sklearn.utils import shuffle

# train paths for panorama images, edge and corner maps
imgs_path = 'PanoContext/train/images/'
edges_path = 'PanoContext/train/edges/'
corners_path = 'PanoContext/train/juncs/'

train_data = os.listdir(imgs_path)

# val paths for panorama images, edge and corner maps
val_imgs_path = 'PanoContext/val/images/'
val_edges_path = 'PanoContext/val/edges/'
val_corners_path = 'PanoContext/val/juncs/'

val_data = os.listdir(val_imgs_path)

n_train = len(train_data)
n_valid= len(val_data)

# shuffling training data
train_data = shuffle(train_data)
batch_size = 8

# data generator for training samples
def data_generator(data, train= True, batch_size=32, number_of_batches=None):
 counter = 0

 #training parameters
 train_w , train_h = 1024,512
 if train:
    path_img = imgs_path
    path_edge = edges_path
    path_corner = corners_path
 else:
    path_img = val_imgs_path
    path_edge = val_edges_path
    path_corner = val_corners_path
 while True:
  idx_start = batch_size * counter
  idx_end = batch_size * (counter + 1)
  x_batch = []
  y_batch = []
  y_batch2 = []
  for file in data[idx_start:idx_end]:
   # read panorama image 
   img = cv2.imread(path_img+file)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   # read edge map
   edge = cv2.imread(path_edge+file)
   edge = cv2.cvtColor(edge, cv2.COLOR_BGR2RGB)
   # read corner map
   corner = cv2.imread(path_corner+file, 0)
   
   # random values for augmentation
   rand_hflip = random.randint(0, 1)
   rand_brightness = random.randint(0, 1)
   # apply random horizontal flip
   if (rand_hflip == 1 and train):
       img = cv2.flip(img, 1)
       edge = cv2.flip(edge, 1)
       corner = cv2.flip(corner, 1)
   # apply random brightness to the input image
   if (rand_brightness == 1 and train):
       val = random.uniform(-1, 1) * 30
       img = img + val
       img = np.clip(img, 0, 255)
       img = np.uint8(img)
   
   img = preprocess_input(img)
   x_batch.append(img) 
   y_batch.append(edge)
   y_batch2.append(corner)
   
  counter += 1
  x_train = np.array(x_batch)
  y_train = np.array(y_batch)
  y_train2 = np.array(y_batch2)
  yield x_train, [y_train , y_train2]
  if (counter == number_of_batches):
        counter = 0

#read reference edge map
ref_img = tf.io.read_file('ref_edge.png')
ref_img = tf.io.decode_png(ref_img)
ref_img = tf.cast(ref_img, tf.float32)
ref_img = ref_img[tf.newaxis,...]
ref_img = tf.tile(ref_img, [batch_size,1,1,1])
print(ref_img.shape)

#read reference corner map
ref_img2 = tf.io.read_file('ref_corner.png')
ref_img2 = tf.io.decode_png(ref_img2)
ref_img2 = tf.image.rgb_to_grayscale(ref_img2)
ref_img2 = tf.cast(ref_img2, tf.float32)
ref_img2 = ref_img2[tf.newaxis,...]
ref_img2 = tf.tile(ref_img2, [batch_size,1,1,1])
print(ref_img2.shape)

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

#compile the model with Adam's optimizer with weight decay, huber loss, and loss weights of 0.75 and 0.25
model.compile(optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay = 0.0001), 
              loss = ['huber_loss','huber_loss'], loss_weights= [0.75, 0.25])

model.summary()

filepath="E:/checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-{val_loss:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
callbacks_list = [checkpoint]

#train the model for 500 epochs using the data generator built above
model.fit(data_generator(train_data, True, batch_size, number_of_batches= n_train // batch_size),
            steps_per_epoch=max(1, n_train//batch_size), initial_epoch = 0,
            validation_data= data_generator(val_data, False, batch_size, number_of_batches= n_valid // batch_size),
            validation_steps=max(1, n_valid//batch_size),
            epochs=500,
            callbacks=callbacks_list)