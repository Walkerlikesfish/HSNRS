"""
ERN - assemble UAV patches

Liu Yu, Liu Shuo

This module serve to assemble the patches together for later visualisation and post-processing

The procedure is like:

"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import argparse
import sys
import scipy.io
import scipy.misc
import test_performance as tp
import os


import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--name', help="string, input the model name")
parser.add_argument("-t", "--targ", help="target folder")
parser.add_argument("-i", "--img", help="the img to infer")
args = parser.parse_args()


# ------------------------------------------------------- #
# Modify the parameters and data path
# ------------------------------------------------------- #
patch_size = [256, 256]
labels = [0, 1, 2, 3, 4, 5]
N_class = 6

# overlap rate
overlap = [0.5, 0.5]

# assemble flag of edge
assemble_edge_flag = 0 # flag to assemble the edge or not | 0:not 1:save

# raw root, for reading the raw image size (w,h)
raw_root = '/media/lh/D/Data/Partion1/test/label/'

# the infered patches folder, the assemble results folder

rgb_infer_root = args.targ[0:-1] + '_infer/'
assemble_root = args.targ[0:-1] + '_assemble/'
# rgb_infer_root = '/media/lh/D/Data/Part1_split_test_overlap_50/' + 'tag_infer_' + args.name +  '/'
# assemble_root = '/media/lh/D/Data/Partion1/UAV_results/' + args.name + '_OL' + '/'   

if not os.path.exists(assemble_root):
    os.mkdir(assemble_root)
if assemble_edge_flag == 1 and not os.path.exists(assemble_edge_root):
    os.mkdir(assemble_edge_root)
# ------------------------------------------------------- #

def xf_UAV_assemble_OverLap():
    """
    This function serve to assemble the inference result with overlap, to carry out overlap inference.

    method: select most vote for the class

    """
    fname = args.img
    img_id = fname[-9:-4] # 5 digits number
    # read the original image
    im = Image.open(fname, 'r')
    # calculate the size of image
    n_im_size = im.size
    n_w = int(n_im_size[0]/patch_size[0]/(1-overlap[0]) + 1)
    n_h = int(n_im_size[1]/patch_size[1]/(1-overlap[1]) + 1)
    # build the new empty image
    re_w = int(patch_size[0] + n_w*patch_size[0]*(1-overlap[0]))
    re_h = int(patch_size[1] + n_h*patch_size[1]*(1-overlap[1]))
    im_re = np.zeros((N_class, re_h, re_w))
    np_re = np.zeros((N_class, re_h, re_w))
    np_re_c = np.zeros((N_class, re_h, re_w))

    s_pfile_root = 'UAV_' + img_id
    pre_id = 0
    # iterate the height

    #print(im_re.shape)
    #print(fname)

    iu = 0
    for ih in range(0, n_h):
        il = 0
        np_il = 0
        for iw in range(0, n_w):  
            # use the data_root to indicate where the data is stored
            fname = rgb_infer_root + s_pfile_root + '_' + str(pre_id).zfill(4) + str(0) + '.png'
            fname_npy = rgb_infer_root + s_pfile_root + '_' + str(pre_id).zfill(4) + str(0) + '.npy'

            pre_id += 1
            im_pre = Image.open(fname)
            im_pre_np = np.asarray(im_pre)
            np_pre = np.load(fname_npy)
            # [most vote count here]
            for iih in range(0, patch_size[0]):
                for iiw in range(0, patch_size[1]):
                    itag = im_pre_np[iih, iiw]
                    #print([itag, iu+iih, il+iiw])
                    im_re[itag, iu+iih, il+iiw] += 1
            # [np array assemble here]
            np_re[:, iu:iu + patch_size[0], il:il + patch_size[1]] += np_pre
            np_re_c[:, iu:iu + patch_size[0], il:il + patch_size[1]] += 1   # adding up the counter
            il = int(il + patch_size[1] - overlap[1] * patch_size[1])
        iu = int(iu + patch_size[0] - overlap[0] * patch_size[0])
    
    
    np_re_sum = np_re / np_re_c
    np_re_sum_c = np_re_sum[:, 0:n_im_size[1], 0:n_im_size[0]]

    im_re_max = np.argmax(np_re_sum, axis=0)
    im_re_c = im_re_max[0:n_im_size[1], 0:n_im_size[0]]
    rgb_save = assemble_root + s_pfile_root + '.png'
    scipy.misc.toimage(im_re_c, cmin=0, cmax=255).save(rgb_save)

if __name__ == '__main__': 
    # without overlap
    xf_UAV_assemble_OverLap()
    


    
