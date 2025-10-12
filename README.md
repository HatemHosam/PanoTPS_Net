# PanoTPS-Net
PanoTPS-Net: Panoramic Room Layout Estimation via Thin Plate Spline Transformation

This is the implementation of PanoTPS-Net: Estimating Room Layout From Single Panorama Images Using Thin Plate Spline Transformation Network

The model was implemented using TensorFlow.

The version of libraries:

Tensorflow-gpu == '2.10.1'
Tensorflow_addons == '0.19.0'
OpenCV == '4.8.0'
NumPy == '1.23.5'

The implementation of the Thin-plate spline (TPS) transformer implementation is based on the following Github-Repo:
https://github.com/dantkz/spatial-transformer-tensorflow

The model was implemented using TensorFlow.
The version of libraries:

Tensorflow-gpu == '2.10.1'
Tensorflow\_addons == '0.19.0'
OpenCV == '4.8.0'
NumPy == '1.23.5'
Scipy == '1.9.1'


The implementation of the Thin-plate spline (TPS) transformer implementation is based on the following Github-Repo:
https://github.com/dantkz/spatial-transformer-tensorflow

For the cuboid room layout estimation, there are two training configurations. First, training on the PanoContext training set and whole Stanford-2D3D and testing on the PanoContext test set. Second, training on Stanford-2D3D training set and whole PanoContext and testing on Stanford-2D3D test set.
The panoContext and Stanford2D3D dataset split and the post-processing utilized are the same as Layout-Netv2  adopted from their official Github repo:
https://github.com/zouchuhang/LayoutNetv2

To train PanoTPS-Net, the dataset should be downloaded from the above link. and the path for images, edges, and corners should be provided in the train.py.
To get the best model fitting in training, it is recommended to train the model for 250 epochs with a learning rate of 0.001, and weight decay of 0.0001. For the second 250 epochs, it is better to change the learning rate to 0.0001 and weight decay of 0.00001.

To test the model, the trained weights should be loaded, and the edge and corner maps should be predicted. Using the post-proc.py code from LayoutNetv2 to get the fine sharp edge map and to get the corner points. LayoutNetv2 also provides a code to render the 3D representation of the room based on the predicted edge and corner maps.

To train PanoTPS-Net on Matterport3DLayout for the general room layout estimation, the same configuration are used as mentioned for the cuboid room estimation (in train_general_layout.py). In testing, the extra stage of the corner map post-processing is applied to the edge and corner maps predictions of PanoTPS-Net. The test script for the general layout estimation is test_general_layout.py.
