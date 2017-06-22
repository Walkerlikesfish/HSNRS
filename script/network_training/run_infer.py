import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc
import test_performance as tp
import os
import sys
import time

caffe_bvlc_root = '/home/yuliu/Documents/Tread/caffe/'
caffe_SegNet_root = '/home/yuliu/Documents/Tread/SegNet/caffe-segnet/'
caffe_SegNetRS_root = '/home/supercaffe0/0RSseg/DeepNetsForEO/caffe/'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="indicate the model path")
parser.add_argument("-w", "--weight", help="indicate if it is training from a existing weight")
parser.add_argument("-i", "--idgpu", help="the id of the GPU used to infer")
args = parser.parse_args()

import ConfigParser
Config = ConfigParser.ConfigParser()
# Configure here to set the data config files
Config.read("../setting_data_train.ini")
caffe_root = Config.get('Caffe', 'CaffeRoot')

sys.path.insert(0, caffe_root + 'python')

import caffe

c_proto = args.model
c_model = args.weight
# load net
caffe.set_mode_gpu()
caffe.set_device(int(args.idgpu))
net = caffe.Net(c_proto,
                c_model,
                caffe.TEST)


# [M]odify the mean if you need
#m_mu_rgb = np.load('../data/VH/mean_rgb.npy')
m_mu_rgb = np.array([73.83527, 74.98086, 108.47272])
#m_mu_ndsm = np.load('../data/VH/mean_ndsm.npy')
m_mu_ndsm = np.array([128.77069])

# [M]odify the data source folder if you need
data_root = '/home/yuliu/Documents/0RSNet/data/VH/dist/testA1/'
image_root = data_root + 'rgb/'
tag_root = data_root + 'tag_nob_png/'
ndsm_root = data_root + 'ndsm/'
result_root = data_root + 'tmp_result/'

# Define the number of labels
nlabels = 6
labels = [0, 1, 2, 3, 4, 5]

save_image = 1  # flag to save the infer result or not | 0:not 1:save

mat_confusion_sum = np.zeros([nlabels, nlabels])

F_score_sum = np.zeros((6, 1))
acc_score_sum = 0
prec_score_sum = np.zeros((6, 1))
cnt = np.zeros((6, 1))
print 'Network initialised...'

num_files = len([f for f in os.listdir(image_root)
                if os.path.isfile(os.path.join(image_root, f))])
print 'infering total: ' + str(num_files) + ' images...'

cnt_time = 0
sum_time = 0

for root, dirs, files in os.walk(image_root):
    for onefile in files:
        fname = image_root + onefile
        if onefile.startswith('VH_'):
            im = caffe.io.load_image(fname)
            im_rgb = im
            start_time = time.time()
            # TODO: Damn it there is a bug in the file name, anyway just remember 0:-5 for A2-3 | 0:-4 for A1
            im_tag = Image.open(tag_root + onefile[0:-5] + '.png')
            # transform the input image
            transformer = caffe.io.Transformer({'data_rgb': net.blobs['data_rgb'].data.shape})
            transformer.set_transpose('data_rgb', (2, 0, 1))  # move image channels to outermost dimension
            transformer.set_mean('data_rgb', m_mu_rgb)  # subtract the dataset-mean value in each channel
            transformer.set_raw_scale('data_rgb', 255)  # rescale from [0, 1] to [0, 255]
            transformer.set_channel_swap('data_rgb', (2, 1, 0))  # swap channels from RGB to BGR
            transformed_image = transformer.preprocess('data_rgb', im)
            in_ = transformed_image
            # shape for input (data blob is N x C x H x W), set data
            net.blobs['data_rgb'].reshape(1, *in_.shape)
            net.blobs['data_rgb'].data[...] = in_
            # Load the ndsm image
            fname = ndsm_root + onefile[0:-4] + '.png'
            im = caffe.io.load_image(fname, color=False)
            transformer = caffe.io.Transformer({'data_ndsm': net.blobs['data_ndsm'].data.shape})
            transformer.set_mean('data_ndsm', m_mu_ndsm)  # subtract the dataset-mean value in each channel
            transformer.set_transpose('data_ndsm', (2, 0, 1))  # move image channels to outermost dimension
            transformer.set_raw_scale('data_ndsm', 255)  # rescale from [0, 1] to [0, 255]
            transformed_image = transformer.preprocess('data_ndsm', im)
            in_ = transformed_image
            # shape for input (data blob is N x C x H x W), set data
            net.blobs['data_ndsm'].reshape(1, *in_.shape)
            net.blobs['data_ndsm'].data[...] = in_
            # run net and take argmax for prediction
            
            # wraped by timer
            
            net.forward()
            end_time = time.time()
            sum_time += (end_time-start_time)
            cnt_time += 1

            out = net.blobs['score'].data[0].argmax(axis=0)

            if save_image == 1:
                # save the class as png image
                save_fname = result_root + onefile[0:-4] + '.png'
                scipy.misc.toimage(out, cmin=0, cmax=255).save(save_fname)
                # save the score in the same location same name as npy
                save_fname = result_root + onefile[0:-4] + '.npy'
                d_score = net.blobs['score'].data[0]
                np.save(save_fname, d_score)

            out = np.array(out)
            tag_arr = np.array(im_tag)

            mat_confusion = tp.x_make_confusion_mat(tag_arr, out, labels, 255)
            mat_confusion_sum = mat_confusion + mat_confusion_sum
            # should not calculate in this way...

mat_prec = tp.x_calc_prec(mat_confusion_sum, labels)
mat_recall = tp.x_calc_recall(mat_confusion_sum, labels)
mat_fscore = tp.x_calc_f1score(mat_prec, mat_recall, labels)

s_prec = tp.x_calc_over_prec(mat_confusion_sum, labels)
s_recall = tp.x_calc_over_recall(mat_confusion_sum, labels)
s_fscore = tp.x_calc_over_fscore(s_prec, s_recall)

s_acc = tp.x_calc_over_acc(mat_confusion_sum, labels)

print "F1_score = " + str(mat_fscore)
print "overall accuracy = " + str(s_acc)
print "precision = " + str(mat_prec)
print "Confusion Matrix:"
print mat_confusion_sum
print "average time consuming per tile:"
print sum_time/cnt_time
print "total time for 5 test images:"
print sum_time
