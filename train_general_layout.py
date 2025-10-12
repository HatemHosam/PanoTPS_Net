import os
from xml.etree import ElementTree
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from spatial_transformer import ElasticTransformer, ProjectiveTransformer
from tensorflow.keras.applications.xception import preprocess_input, Xception
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import *
import random
import scipy.io
from sklearn.utils import shuffle

imgs_path = 'E:/panorama_layout/mp3d/image/'
edges_path = 'E:/panorama_layout/mp3d/edge_maps/'
corners_path = 'E:/panorama_layout/mp3d/corner_maps/'

with open('E:/panorama_layout/mp3d/split/train.txt') as file:
    train_data = file.readlines()

with open('E:/panorama_layout/mp3d/split/val.txt') as file:
    val_data = file.readlines()

n_train = len(train_data)
n_valid= len(val_data)

train_data = shuffle(train_data)
batch_size = 8


def data_generator(data, train= True, batch_size=32, number_of_batches=None):
 counter = 0

 #training parameters
 train_w , train_h = 1024,512
 while True:
  idx_start = batch_size * counter
  idx_end = batch_size * (counter + 1)
  x_batch = []
  y_batch = []
  y_batch2 = []
  for file in data[idx_start:idx_end]:
   img = cv2.imread(imgs_path+file.split(' ')[0]+'_'+file.split(' ')[1].split('\n')[0]+'.png')
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   #img = cv2.resize(img, (train_w , train_h))
   edge = cv2.imread(edges_path+file.split(' ')[0]+'_'+file.split(' ')[1].split('\n')[0]+'.png')
   edge = cv2.cvtColor(edge, cv2.COLOR_BGR2RGB)
   corner = cv2.imread(corners_path+file.split(' ')[0]+'_'+file.split(' ')[1].split('\n')[0]+'.png', 0)
   
   rand_hflip = random.randint(0, 1)
   rand_brightness = random.randint(0, 1)
   
   if (rand_hflip == 1 and train):
       img = cv2.flip(img, 1)
       edge = cv2.flip(edge, 1)
       corner = cv2.flip(corner, 1)

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

ref_img = tf.io.read_file('PC_0016_ref_edge.png')
ref_img = tf.io.decode_png(ref_img)[:,:,:3]
ref_img = tf.cast(ref_img, tf.float32)
ref_img = ref_img[tf.newaxis,...]
ref_img = tf.tile(ref_img, [batch_size,1,1,1])
print(ref_img.shape)

ref_img2 = tf.io.read_file('PC_0016_ref_corner.png')
ref_img2 = tf.io.decode_png(ref_img2)#[:,:,:3]
ref_img2 = tf.image.rgb_to_grayscale(ref_img2)
ref_img2 = tf.cast(ref_img2, tf.float32)
ref_img2 = ref_img2[tf.newaxis,...]
ref_img2 = tf.tile(ref_img2, [batch_size,1,1,1])
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


model.compile(optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay = 0.0001), loss = ['huber_loss','huber_loss'], loss_weights= [0.75, 0.25])

model.summary()

filepath="E:/panorama_layout/mp3d/weights_stn_MXception_edges+corners_64pt_mp3d/weights-improvement-{epoch:02d}-{loss:.4f}-{val_loss:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
callbacks_list = [checkpoint]

model.fit(data_generator(train_data, True, batch_size, number_of_batches= n_train // batch_size),
            steps_per_epoch=max(1, n_train//batch_size), initial_epoch = 0,
            validation_data= data_generator(val_data, False, batch_size, number_of_batches= n_valid // batch_size),
            validation_steps=max(1, n_valid//batch_size),
            epochs=500,
            callbacks=callbacks_list)