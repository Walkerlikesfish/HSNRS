"""

This script is used to adapt the tiff tag image to single channel .png tag image

name rule:
    VH_XX_AAAAB: XX from top split at AAAA,
     B: 1 - stands for origin split, 2 - flip, 3 - rotation,

usage:
    1) run the x_adapt_tags(), setting the data_root and all the path, to adapt the
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import scipy.misc
import os
from multiprocessing.dummy import Pool as ThreadPool
import cv2

# ------------------------------------------------------------- #
# BASIC original image folder settings
#
# ! set the data root first !
data_root = '/home/yuliu/Documents/0RSNet/data/VH/dist/testA3/'
rgb_folder = 'top/'
tag_folder = 'gts_for_participants/'
tag_folder2 = 'gts_no_boarder/'
# ------------------------------------------------------------- #

# ------------------------------------------------------------- #
# Source & Target folder
# without boarder
t_tag_folder2 = 'tag_nob/'
t_tag_png_folder2 = 'tag_nob_png/'
# with boarder
t_tag_folder = 'tag/'
t_tag_png_folder = 'tag_png/'
# ------------------------------------------------------------- #

t_rgb_folder = 'split/rgb/'

training_set = [1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37, 11, 15, 28, 30, 34]

# hard written index: it is only 6 class...
# reference file VH_class.txt
# [S]elect the corresponding index of colors

# the following is for VH
d_index = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]

# the following option is for MA
#d_index = [(0, 0, 0), (255, 0, 0)]

# index fo
# d_index = [(255, 255, 255),     # background -> label 0 don't count
#            (0, 0, 0),           # Roads
#            (100, 100, 100),     # Buildings
#            (0, 125, 0),         # Trees
#            (0, 255, 0),         # Grass
#            (150, 80, 0),        # Bare Soil
#            (0, 0, 150),         # Water
#            (255, 255, 0),       # Railways
#            (150, 150, 255)]     # Swimming Pools

# Initialisation of the set
# training_set = [1, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 17, 18, 20]
# test_set = [2, 6, 12, 15, 16, 19]


def x_adapt_one(im, tt_path):
    if not os.path.isfile(tt_path):
        im_arr = np.array(im)

        weight = im_arr.shape[1]
        height = im_arr.shape[0]
        t_im_arr = np.zeros([height, weight], dtype=np.uint8)

        for ix in xrange(height):
            for iy in xrange(weight):
                p_cur = tuple(im_arr[ix][iy])
                if (p_cur in d_index):
                    t_im_arr[ix][iy] = d_index.index(p_cur)
                else:
                    t_im_arr[ix][iy] = 255  # 255 mark as ignored

        Image.fromarray(t_im_arr).save(tt_path)
    else:
        pass


def xd_adapt_tags_patches(cind):
    '''
    Adapt the color tag to integer tag map file
    t_tag_folder: the source folder of image with boarder
    t_tag_folder2: the source folder of image without boarder
    :return:
    '''
    folder_root = data_root+t_tag_folder
    for root, dirs, files in os.walk(folder_root):
        for onefile in files:
            fname = folder_root + onefile
            if onefile.startswith('VH_'+str(cind)):
		t_fname = data_root + t_tag_png_folder2 + onefile[0:-5] + '.png'                
		if not os.path.isfile(t_fname):
			t_fname = data_root + t_tag_png_folder + onefile[0:-5] + '.png'
		   	im = Image.open(fname)
		        x_adapt_one(im, t_fname)
		        t_fname = data_root + t_tag_png_folder2 + onefile[0:-5] + '.png'
		        fname = data_root + t_tag_folder2 + onefile
		        im = Image.open(fname)
		        x_adapt_one(im, t_fname)


def xf_adapt_test_tags(cind):
    '''
    Adapt the color palett colored tag image to grayscale(number based) image

    if only one type of tag is used, use this function

    t_tag_folder2: used for source image folder
    t_tag_png_folder2: used for the destination image folder
    :return:
    '''
    print 'deal with' + str(cind) + 'tile...'
    gts_root_path = '../PD/raw/tag/'
    s_prefix = 'top_mosaic_09cm_area'
    fname = gts_root_path + s_prefix + str(cind) + '.tif'
    fname_tar = '../../misc/' + 'top_gt_withboarder_09cm_area_' + str(cind)+ '.png'
    im_gt = Image.open(fname)
    x_adapt_one(im_gt, fname_tar)


def xf_adapt_PD_tags(cind):
    data_root = '../PD/raw/'
    tag_folder = 'tag/'
    target_folder = 'tag_png/'
    folder_root = data_root + tag_folder
    for root, dirs, files in os.walk(folder_root):
        for onefile in files:
            fname = folder_root + onefile
            if fname.endswith(".tif") and onefile.startswith('top_potsdam_'+str(cind)):
                print 'adapting tag file ' + onefile
                t_fname = data_root + target_folder + onefile[0:-4] + '.png'
                im = Image.open(fname)
                x_adapt_one(im, t_fname)


def xf_adapt_mt(nmt):
    """
    Multi-threading to adapt tags
    :param nmt: the number of threads you wish to start
    :return:
    """
    # simple multi-threading
    pool = ThreadPool(nmt)
    pool.map(xf_adapt_test_tags, test_set)


if __name__ == '__main__':
    xd_adapt_tags_patches(3)
