import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import cv2
import time
#import liblib.so
from PIL import Image
from keras.preprocessing.image import *
from keras.models import load_model
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input
import pdb
from models import *
import ctypes
import keras2caffe

import sys
import os

caffe_root='../caffeProjects/caffe'
sys.path.insert(0,os.path.join(caffe_root,'python'))
import caffe
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
caffe.set_mode_cpu()
def inference():
    model_name = 'AtrousFCN_DeResnet50_asUnet'
    weight_file = 'checkpoint_weights_1.hdf5'
    model_input_size=(16,32)
    batch_shape = (1, ) + model_input_size + (3, )
    model_weight_path = os.path.join('Models',model_name, weight_file)
    model = globals()[model_name](batch_shape=batch_shape, input_shape=(model_input_size[0], model_input_size[1], 3))
    model.load_weights(model_weight_path, by_name=True)
    # model.summary()
    end_layer_name='activation_1'
    test_keras_model=Model(input=model.input,output=model.get_layer(end_layer_name).output)
#    test_keras_model.summary()
    caffe_model_dir='caffeModel'
    caffe_net_file= os.path.join(caffe_model_dir,'test_model_deploy.prototxt')
    caffe_params_file=os.path.join(caffe_model_dir,'test_model.caffemodel')
    keras2caffe.convert(test_keras_model, caffe_net_file, caffe_params_file)

    print('model transformation done')

    caffe_model = caffe.Net(caffe_net_file,caffe_params_file,caffe.TEST)
    #input_data=np.load('midResult/input_data.npy')
    input_data=np.random.random(model.input.shape)
    
    caffe_model.blobs['data'].data[...] = np.transpose(input_data,(0,3,1,2))      #执行上面设置的图片预处理操作，并将图片载入到blob中

    caffe_out=caffe_model.forward()
    output=caffe_out['conv1']

    output=np.transpose(output,(0,2,3,1))
    print(output)
    
    keras_out=test_keras_model.predict(input_data,batch_size=1)
    print(keras_out)

if __name__ == '__main__':
    inference()
    
