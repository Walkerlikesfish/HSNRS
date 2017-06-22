# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
import ConfigParser
Config = ConfigParser.ConfigParser()
# Configure here to set the data config files
Config.read("../setting_data_train.ini")
caffe_root = Config.get('Caffe', 'CaffeRoot')
import os
import sys


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="show the graph of the net",
                    action="store_true")
parser.add_argument("-m", "--model", help="indicate the model path")
args = parser.parse_args()

if args.image:
    print "Saving and showing graph"

sys.path.insert(0, caffe_root + 'python')
print args.model

import caffe
from caffe import draw
from caffe.proto import caffe_pb2

caffe.set_device(int(1))
caffe.set_mode_gpu()

model_segbasic_naive_def = '/home/yuliu/Documents/Tread/SegNet/VH_Models/segnet_inference.prototxt'
model_fcn_alex_naive_def = '/home/yuliu/digits/digits/jobs/20161215-222500-70ef/train_val.prototxt'
model_fcn_alex_naive_def = './tmp_net/segnet_inference.prototxt'
model_fcn_alex_append_def = '/home/yuliu/Documents/Tread/scripts/tmp_net/train_val_b1.prototxt'

model_def = args.model

net = caffe.Net(model_def,      # defines the structure of the model
                caffe.TRAIN)     # use test mode (e.g., don't perform dropout)

# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

# Draw the net structure save to tmp.png
if args.image:
    os.system("python /home/supercaffe0/0RSseg/DeepNetsForEO/caffe/python/draw_net.py --rankdir TB " + model_def + ' ./tmp.png')
    Image.open('./tmp.png').show()


