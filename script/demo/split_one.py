import numpy as np
from PIL import Image
import os
import ConfigParser
# Configure here to set the data config filesimport numpy as np
from PIL import Image
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img", help="indicate the image to split")
parser.add_argument("-t", "--targ", help="indicate the target folder to save the splitted image")
args = parser.parse_args()

def x_calc_lucorners(im, patch_size, overlap_size=[0, 0]):
    '''

    :param im:
    :param patch_size:
    :param overlap_size: [0, Patchsize[*]] indicating how much is the overlaping
    :param boarderflag: is the flag marking the way to deal with the boarder
        - set 1(default): to discard the residual pixels
        - set 2 : to reflect along the boarder
    :return:
        corners: a list including the corner coordinates [x,y]
    '''
    im_arr = np.array(im)
    # shape is always height first
    (im_height, im_width) = (im_arr.shape[0], im_arr.shape[1])
    l_x = []
    l_y = []
    cur_x = 0
    cur_y = 0
    while cur_x < im_height:
        while cur_y< im_width:
            l_x.append(cur_x)
            l_y.append(cur_y)
            cur_y = cur_y + patch_size[1] - overlap_size[1]

        cur_x = cur_x + patch_size[0] - overlap_size[0]
        cur_y = 0
    return [l_x, l_y]


def x_one_split(l_cor, patch_size, origin_image):
    '''
    split one image PATCH at l_cor position

    :param l_cor: the corner list, indicating the up left corner
    :param patch_size:
    :param origin_image:
    :return:
     split image
    '''
    box = (l_cor[1], l_cor[0], l_cor[1]+patch_size[1], l_cor[0]+patch_size[0])
    c_im = origin_image.crop(box)
    return c_im


def x_image_split(l_corners, patch_size, origin_image, root_name):
    '''
    Split one image PATCH with augmentation: flip H, flip V, rotation -> x8

    :param l_corners: corner list
    :param patch_size:
    :param origin_image: input image tile
    :param root_name: save root folder name
    :return:
    '''
    n_corners = len(l_corners[0])
    for i in xrange(n_corners):
        fname = root_name + str(i).zfill(4) + str(0) + '.png'
        l_cor = [l_corners[0][i], l_corners[1][i]]
        c_im = x_one_split(l_cor, patch_size, origin_image)
        c_im.save(fname)
        # data augmentation
        # 1. filp lr
        fname = root_name + str(i).zfill(4) + str(1) + '.png'
        tc_im = c_im.transpose(Image.FLIP_LEFT_RIGHT)
        tc_im.save(fname)
        # 2. flip ud + r90
        fname = root_name + str(i).zfill(4) + str(2) + '.png'
        tc_im = c_im.transpose(Image.FLIP_TOP_BOTTOM)
        tc_im.save(fname)
        fname = root_name + str(i).zfill(4) + str(3) + '.png'
        tc_im = tc_im.rotate(90)
        tc_im.save(fname)
        fname = root_name + str(i).zfill(4) + str(4) + '.png'
        tc_im = tc_im.rotate(180)
        tc_im.save(fname)
        # 4. rotation 90 180 270
        cdx = 5
        for ra in xrange(90, 360, 90):
            fname = root_name + str(i).zfill(4) + str(cdx) + '.png'
            tc_im = c_im.rotate(ra)
            tc_im.save(fname)
            cdx = cdx + 1


def x_image_split_no_aug(l_corners, patch_size, origin_image, root_name):
    '''
    split the origin_image into patch size, with the direction given by l_corners
    WITHOUT augmentation -> for test dataset preprocessing

    :param l_corners: a list containing the spliting strating points
    :param patch_size: the patch size
    :param origin_image: original image which is the tile
    :param root_name: saving folder and prefix
    :return:
    '''
    n_corners = len(l_corners[0])
    for i in xrange(n_corners):
        fname = root_name + str(i).zfill(4) + str(0) + '.png'
        l_cor = [l_corners[0][i], l_corners[1][i]]
        c_im = x_one_split(l_cor, patch_size, origin_image)
        c_im.save(fname)


def x_calc_lucorners(im, patch_size, overlap_size=[0, 0]):
    '''

    :param im:
    :param patch_size:
    :param overlap_size: [0, Patchsize[*]] indicating how much is the overlaping
    :param boarderflag: is the flag marking the way to deal with the boarder
        - set 1(default): to discard the residual pixels
        - set 2 : to reflect along the boarder
    :return:
        corners: a list including the corner coordinates [x,y]
    '''
    im_arr = np.array(im)
    # shape is always height first
    (im_height, im_width) = (im_arr.shape[0], im_arr.shape[1])
    l_x = []
    l_y = []
    cur_x = 0
    cur_y = 0
    while cur_x < im_height:
        while cur_y< im_width:
            l_x.append(cur_x)
            l_y.append(cur_y)
            cur_y = cur_y + patch_size[1] - overlap_size[1]

        cur_x = cur_x + patch_size[0] - overlap_size[0]
        cur_y = 0
    return [l_x, l_y]



def xf_UAV_split_one_no_aug(ps, overlap, fname, targ_folder, prefix='UAV_'):
    '''
    split one image(provided file name) and save to target folder
    the filename just append id string after the original name

    '''
    # modify here change the overlap coefficient
    patch_size = ps
    overlap_size = [int(patch_size[0]*overlap[0]), int(patch_size[1]*overlap[1])]
    # load and split target image to folder
    img_id = fname[-9:-4]
    im = Image.open(fname, 'r')
    l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
    root_name = targ_folder + prefix + img_id + '_'
    x_image_split_no_aug(l_corners, patch_size, im, root_name)


def main():
    patch_size = [256, 256]
    overlap = [0.5, 0.5]
    fname = args.img
    targ_folder = args.targ
    xf_UAV_split_one_no_aug(ps=patch_size, overlap=overlap, fname=fname, targ_folder=targ_folder)

'''
test case:
[img] /media/lh/D/Data/Partion1/test/img/10009.bmp
[targ_folder] /media/lh/D/Data/Partion1/0tmp
python split_one.py -i /media/lh/D/Data/Partion1/test/img/10009.bmp -t /media/lh/D/Data/Partion1/0tmp
'''


if __name__ == '__main__':
    main()
    
