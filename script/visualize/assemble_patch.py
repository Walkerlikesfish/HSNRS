"""
RSnet - assemble patches

Yu Liu @ VUB ETRO

This module serve to assemble the patches together for later visualisation and post-processing

The procedure is like:
0) use x_assemble_gt() to assemble together the ground truth

"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import argparse
import sys
import scipy
import scipy.misc
import test_performance as tp

# ------------------------------------------------------- #
# Initialisation for parameters
test_set = [11, 15, 28, 30, 34]
patch_size = [256, 256]
labels = [0, 1, 2, 3, 4, 5]
N_class = 6
# reference tile path
ref_data_path = '../data/VH/raw/'
ref_rgb_path = 'top/'
# ------------------------------------------------------- #
# Setting the paths [M]
# please modify the paths for different tasks
data_root = '../data/VH/test/setA1/'
result_root = 'tmp_result/'
tile_result_root = '../misc/'
# ------------------------------------------------------- #


def xf_assemble_one(ind, data_root):
    """
    assemble the ind'th image to the original size

    - assuming no overlap

    :param ind:
    :param data_root: the folder path of the tmp result, say the patches
    :return:
    """
    # load original image to infer the size of the image
    s_folder_root = ref_data_path + ref_rgb_path
    fname = s_folder_root + 'top_mosaic_09cm_area' + str(ind) + '.tif'
    im = Image.open(fname)
    n_im_size = im.size
    n_w = n_im_size[0] / patch_size[0] + 1
    n_h = n_im_size[1] / patch_size[1] + 1
    im_re = np.zeros((n_h * patch_size[0], n_w * patch_size[1]))

    s_pfile_root = 'VH_' + str(ind).zfill(2) + '_'
    pre_id = 0
    # iterate the height
    for ih in range(0, n_h):
        for iw in range(0, n_w):
            # use the data_root to indicate where the data is stored
            fname = data_root + s_pfile_root + str(pre_id).zfill(4) + str(0) + '.png'
            pre_id += 1
            im_pre = Image.open(fname)
            im_pre_np = np.asarray(im_pre)
            il = patch_size[1] * iw
            iu = patch_size[0] * ih
            im_re[iu:iu + patch_size[0], il:il + patch_size[1]] = im_pre_np
    im_re_c = im_re[0:n_im_size[1], 0:n_im_size[0]]
    return im_re_c


def xf_assemble_with_overlap(ind, root_path, overlap):
    """
    This function serve to assemble the inference result with overlap, to carry out overlap inference.

    method: select most vote for the class

    :param ind: the id of the big tile
    :param root_path: the root path of the result folder
    :param overlap: overlap size
    :return:
    """
    # load original image to infer the size of the image
    s_folder_root = ref_data_path + ref_rgb_path
    fname = s_folder_root + 'top_mosaic_09cm_area' + str(ind) + '.tif'
    im = Image.open(fname)
    n_im_size = im.size
    n_w = int(n_im_size[0]/patch_size[0]/(1-overlap[0]) + 1)
    n_h = int(n_im_size[1]/patch_size[1]/(1-overlap[1]) + 1)
    # TODO:[BUG] in counter:
    if ind == 30:
        n_w += 1
    #print n_im_size
    re_w = int(patch_size[0] + n_w*patch_size[0]*(1-overlap[0]))
    re_h = int(patch_size[1] + n_h*patch_size[1]*(1-overlap[1]))
    im_re = np.zeros((N_class, re_h, re_w))

    s_pfile_root = 'VH_' + str(ind).zfill(2) + '_'
    pre_id = 0
    # iterate the height
    iu = 0
    for ih in range(0, n_h):
        il = 0
        for iw in range(0, n_w+1):
            # use the data_root to indicate where the data is stored
            fname = root_path + s_pfile_root + str(pre_id).zfill(4) + str(0) + '.png'
            pre_id += 1
            im_pre = Image.open(fname)
            im_pre_np = np.asarray(im_pre)
            for iih in range(0, patch_size[0]):
                for iiw in range(0, patch_size[1]):
                    itag = im_pre_np[iih, iiw]
                    im_re[itag, iu+iih, il+iiw] += 1
            il = il + patch_size[1] - overlap[1] * patch_size[1]
        iu = iu + patch_size[0] - overlap[0] * patch_size[0]
    im_re_max = np.argmax(im_re, axis=0)
    im_re_c = im_re_max[0:n_im_size[1], 0:n_im_size[0]]
    # return both score and the class map
    return im_re, im_re_c


def xf_assemble_with_overlap_ss(ind, root_path, overlap):
    """
        This function serve to assemble the inference result with overlap, to carry out overlap inference.

        method: sum up and average the scores

        :param ind: the id of the big tile
        :param root_path: the root path of the result folder
        :param overlap: overlap size
        :return:
        """
    # load original image to infer the size of the image
    s_folder_root = ref_data_path + ref_rgb_path
    fname = s_folder_root + 'top_mosaic_09cm_area' + str(ind) + '.tif'
    im = Image.open(fname)
    n_im_size = im.size
    n_w = int(n_im_size[0] / patch_size[0] / (1 - overlap[0]) + 1)
    n_h = int(n_im_size[1] / patch_size[1] / (1 - overlap[1]) + 1)
    # TODO:[BUG] in counter:
    if ind == 30:
        n_w += 1
    # print n_im_size
    re_w = int(patch_size[0] + n_w * patch_size[0] * (1 - overlap[0]))
    re_h = int(patch_size[1] + n_h * patch_size[1] * (1 - overlap[1]))
    # initialise the container
    im_re = np.zeros((N_class, re_h, re_w))
    im_re_c = np.zeros((N_class, re_h, re_w)) # counter matrix (helper)

    s_pfile_root = 'VH_' + str(ind).zfill(2) + '_'
    pre_id = 0
    # iterate the height
    iu = 0
    for ih in range(0, n_h):
        il = 0
        for iw in range(0, n_w+1): # TODO: 
            # use the data_root to indicate where the data is stored
            fname = root_path + s_pfile_root + str(pre_id).zfill(4) + str(0) + '.npy'
            pre_id += 1
            im_pre = np.load(fname)
            im_re[:, iu:iu + patch_size[0], il:il + patch_size[1]] += im_pre
            im_re_c[:, iu:iu + patch_size[0], il:il + patch_size[1]] += 1   # adding up the counter
            il = il + patch_size[1] - overlap[1] * patch_size[1]
        iu = iu + patch_size[0] - overlap[0]*patch_size[0]
    im_re_sum = im_re / im_re_c
    im_re_class = np.argmax(im_re_sum, axis=0)

    # crop the image
    im_re_c = im_re_class[0:n_im_size[1], 0:n_im_size[0]]
    return im_re_sum,im_re_c


def xh_test_assembl(ind, root_path, overlap):
    s_folder_root = ref_data_path + ref_rgb_path
    fname = s_folder_root + 'top_mosaic_09cm_area' + str(ind) + '.tif'
    im = Image.open(fname)
    n_im_size = im.size
    n_w = int(n_im_size[0] / patch_size[0] / (1 - overlap[0]) + 1)
    n_h = int(n_im_size[1] / patch_size[1] / (1 - overlap[1]) + 1)
    # print n_im_size
    re_w = int(patch_size[0] + (n_w) * patch_size[0] * (1 - overlap[0]))
    re_h = int(patch_size[1] + (n_h) * patch_size[1] * (1 - overlap[1]))
    im_re = np.zeros((re_h, re_w))
    s_pfile_root = 'VH_' + str(ind).zfill(2) + '_'
    pre_id = 0
    # iterate the height
    iu = 0
    for ih in range(0, n_h):
        il = 0
        for iw in range(0, n_w+1):
            # use the data_root to indicate where the data is stored
            fname = root_path + s_pfile_root + str(pre_id).zfill(4) + str(0) + '.png'
            pre_id += 1
            im_pre = Image.open(fname)
            im_pre_np = np.asarray(im_pre)
            im_re[iu:iu + patch_size[0], il:il + patch_size[1]] = im_pre_np
            il = il + patch_size[1] - overlap[1] * patch_size[1]
        iu = iu + patch_size[0] - overlap[0]*patch_size[0]
    #im_re_max = np.argmax(im_re, axis=0)
    im_re_c = im_re[0:n_im_size[1], 0:n_im_size[0]]
    return im_re_c


def xf_assemble_set_ol(s_prefix, data_path, overlap):
    """

    :param s_prefix:
    :param data_path:
    :param overlap
    :return:
    """
    for cind in test_set:
        print 'Assembling the ' + str(cind) + 'th image...Be patient..'
        score, out = xf_assemble_with_overlap_ss(cind, data_path, overlap)
        fname_save = tile_result_root + s_prefix + str(cind) + '.png'
        scipy.misc.toimage(out, cmin=0, cmax=255).save(fname_save)
        np.save(tile_result_root + s_prefix + str(cind) + '.npy', score)
    print 'Finish assemble the files'
    print 'save to: ' + tile_result_root + s_prefix


def xf_assemble_set(s_prefix, data_root):
    """
    Assemble the whole test_set to big tiles
    :param s_prefix:
    :param data_root: specify where the tmp_result is located
    :return:
    """
    for cind in test_set:
        print 'Assembling the'+str(cind)+'th image...Be patient..'
        out = xf_assemble_one(cind, data_root)
        fname_save = tile_result_root + s_prefix + str(cind) + '.png'
        scipy.misc.toimage(out, cmin=0, cmax=255).save(fname_save)
    print 'Finish assemble the files'
    print 'save to: ' + tile_result_root + s_prefix


def xh_assemble_gt():
    """
    Assemble the ground truth set for the evaluation and visualisation use

    :return:
    """
    s_prefix = 'top_gt_09cm_area_'
    gt_path =  '../data/VH/test/setA1/tag_nob_png/'
    xf_assemble_set(s_prefix, gt_path)
    print 'finish assemble the ground truth for test set!'


def xh_test():
    overlap = [0.75, 0.75]
    s_prefix = 'hsnet_ois_75_'
    xf_assemble_set_ol(s_prefix, data_root+result_root, overlap)

if __name__ == '__main__':
    data_path = data_root + result_root
    s_prefix = 'hsnet_only_'
    overlap = [0.75, 0.75]
    xf_assemble_set(s_prefix, data_path)
    # xf_assemble_set_ol(s_prefix, data_path, overlap)
    #xh_assemble_gt()
    # xh_test()
