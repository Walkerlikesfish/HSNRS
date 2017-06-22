"""

name rule:
    VH_XX_AAAAB: XX from top split at AAAA,
     B: 1 - stands for origin split, 2 - flip, 3 - rotation,

usage:
    1) setting the folder as indicated structure (creating the split folder to contain the augmented images)
    2) run the x_VH_split() wiht selected patch size
    3) generate the file index is optional
"""
import numpy as np
from PIL import Image
import os
import ConfigParser
Config = ConfigParser.ConfigParser()
# Configure here to set the data config files
Config.read("../setting_data_train.ini")

training_set = [1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37, 11, 15, 28, 30, 34] # VH training index
test_set = [11, 15, 28, 30, 34] # VH test index


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

# -------------------------------------------------------------------- #
#                  Function to call directly                           #
# -------------------------------------------------------------------- #
def xf_VH_train_split(ps, overlap):
    '''
    Split the VH dataset with spcified patch_size;
    spliting both RGB data and the TAG data
    dave to
    t_rgb_folder and t_tag_folder
    modify the string to save it to different place

    :param ps: a tuple including the [height, width] of the patch
    :param overlap: Define the overlap size [height, width]
    :return:
    '''
    # modify here change the overlap coefficient
    patch_size = ps
    overlap_size = [int(patch_size[0]*overlap[0]), int(patch_size[1]*overlap[1])]

    data_root = Config.get('VH_root', 'data_root')
    rgb_folder = Config.get('VH_root', 'rgb_folder')
    tag_folder = Config.get('VH_root', 'tag_folder')
    dsm_folder = Config.get('VH_root', 'dsm_folder')
    tag_folder2 = Config.get('VH_root', 'tag_nob_folder')

    t_data_root = Config.get('VH_root', 't_data_root')
    t_rgb_folder = t_data_root + Config.get('VH_root', 't_rgb_folder')
    t_tag_folder = t_data_root + Config.get('VH_root', 't_tag_folder')
    t_tag_folder2 = t_data_root + Config.get('VH_root', 't_tag_folder2')
    t_ndsm_folder = t_data_root + Config.get('VH_root', 't_ndsm_folder')

    for idx in training_set:
        # split and augmentation of the RGB data
        folder_root = data_root + rgb_folder
        fname = folder_root + 'top_mosaic_09cm_area' + str(idx) + '.tif'
        im = Image.open(fname)
        l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
        root_name = t_rgb_folder + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split(l_corners, patch_size, im, root_name)
        # split and augmentation of the Tag data
        folder_root = data_root + tag_folder
        fname = folder_root + 'top_mosaic_09cm_area' + str(idx) + '.tif'
        im = Image.open(fname,'r')
        l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
        root_name = t_tag_folder + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split(l_corners, patch_size, im, root_name)
        # split and augmentation of the no-boarder tag data
        folder_root = data_root + tag_folder2
        fname = folder_root + 'top_mosaic_09cm_area' + str(idx) + '_noBoundary' + '.tif'
        im = Image.open(fname, 'r')
        l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
        root_name = t_tag_folder2 + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split(l_corners, patch_size, im, root_name)
        # split ad augmentation of the ndsm images
        folder_root = data_root + dsm_folder
        fname = folder_root + 'dsm_09cm_matching_area' + str(idx) + '_normalized.jpg'
        im = Image.open(fname)
        l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
        root_name = t_ndsm_folder + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split(l_corners, patch_size, im, root_name)


def xf_VH_test_split(ps, overlap):
    '''
    [Reviewed]
    This function is used for split the big tile file into patches
    and save to
    data_root + ttest_xx_folder

    [M]odify the ttest_xx_folder to set the destinations of the saving
    :param: overlap: Specify overlap size [0, 1] (1x2 LIST)
    :param: ps: Specify patch size (1x2 LIST)
    :return:
    '''

    patch_size = ps
    overlap_size = [int(patch_size[0] * overlap[0]), int(patch_size[1] * overlap[1])]

    data_root = Config.get('VH_root', 'data_root')
    rgb_folder = Config.get('VH_root', 'rgb_folder')
    tag_folder = Config.get('VH_root', 'tag_folder')
    dsm_folder = Config.get('VH_root', 'dsm_folder')
    tag_folder2 = Config.get('VH_root', 'tag_nob_folder')

    ttest_root = Config.get('VH_root', 'test_data_root')
    ttest_rgb_folder = ttest_root + Config.get('VH_root', 'test_rgb_folder')
    ttest_tag_folder = ttest_root + Config.get('VH_root', 'test_tag_folder')
    ttest_tag_folder2 = ttest_root + Config.get('VH_root', 'test_tagnob_folder')
    ttest_ndsm_folder = ttest_root + Config.get('VH_root', 'test_ndsm_folder')

    for idx in test_set:
        # split and augmentation of the RGB data
        folder_root = data_root + rgb_folder
        fname = folder_root + 'top_mosaic_09cm_area' + str(idx) + '.tif'
        im = Image.open(fname)
        l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
        root_name = ttest_rgb_folder + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split_no_aug(l_corners, patch_size, im, root_name)
        # split the no-boarder tag data
        folder_root = data_root + tag_folder2
        fname = folder_root + 'top_mosaic_09cm_area' + str(idx) + '_noBoundary' + '.tif'
        im = Image.open(fname, 'r')
        l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
        root_name = ttest_tag_folder2 + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split_no_aug(l_corners, patch_size, im, root_name)
        # split the original tag data
        folder_root = data_root + tag_folder
        fname = folder_root + 'top_mosaic_09cm_area' + str(idx) + '.tif'
        im = Image.open(fname, 'r')
        l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
        root_name = ttest_tag_folder + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split_no_aug(l_corners, patch_size, im, root_name)
        # split the nDSM data
        folder_root = data_root + dsm_folder
        fname = folder_root + 'dsm_09cm_matching_area' + str(idx) +'_normalized.jpg'
        im = Image.open(fname)
        l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
        root_name = ttest_ndsm_folder + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split_no_aug(l_corners, patch_size, im, root_name)


def xf_PD_data_split_train(patch_size, overlap):
    """
    Split the Potsdam dataset with certain overlap and patch size
    :param overlap: [height, weight] in scale [0,1]
    :param patch_size: [height, weight] in pixels
    :return:
    """
    print 'spliting the training data'
    res_root = Config.get('PD_root', 'res_root')
    rgb_folder = 'rgb/'
    ir_folder = 'ir/'
    ndsm_folder = 'ndsm/'
    tag_folder = 'gt/'
    tag_nob_folder = 'gt_nob/'

    n_tile1 = [2, 3, 4, 5, 6, 7]
    n_tile2 = [7, 8, 9, 10, 11]
    prefix = 'top_potsdam_'
    sufix = 'RGB.jpg'
    data_path = res_root + rgb_folder + prefix
    tar_root = Config.get('PD_root', 'tar_root')
    #tar_root = '../PD/split/train/'

    overlap_size = [int(patch_size[0] * overlap[0]), int(patch_size[1] * overlap[1])]
    fid = 0
    for i1 in n_tile1:
        for i2 in n_tile2:
            src_fname = data_path + str(i1) + '_' + str(i2) + '_' + sufix
            if os.path.isfile(src_fname):
                tar_path = tar_root + 'rgb/'
                im = Image.open(src_fname)
                l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
                root_name = tar_path + 'PD_' + str(fid).zfill(2) + '_'
                x_image_split(l_corners, patch_size, im, root_name)

                src_fname = res_root+ndsm_folder+'dsm_potsdam_' + str(i1).zfill(2)+'_'+str(i2).zfill(2)+'_normalized_ownapproach.jpg'
                if not os.path.isfile(src_fname):
                    src_fname = res_root+ndsm_folder+'dsm_potsdam_' + str(i1).zfill(2)+'_'+str(i2).zfill(2)+'_normalized_lastools.jpg'
                tar_path = tar_root + 'ndsm/'
                im = Image.open(src_fname)
                l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
                root_name = tar_path + 'PD_' + str(fid).zfill(2) + '_'
                x_image_split(l_corners, patch_size, im, root_name)

                src_fname = res_root + ir_folder + 'top_potsdam_' + str(i1) + '_' + str(i2) + '_RGBIR.png'
                tar_path = tar_root + 'ir/'
                im = Image.open(src_fname)
                l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
                root_name = tar_path + 'PD_' + str(fid).zfill(2) + '_'
                x_image_split(l_corners, patch_size, im, root_name)

                src_fname = res_root + tag_folder + 'top_potsdam_' + str(i1) + '_' + str(i2) + '.png'
                tar_path = tar_root + 'tag/'
                im = Image.open(src_fname)
                l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
                root_name = tar_path + 'PD_' + str(fid).zfill(2) + '_'
                x_image_split(l_corners, patch_size, im, root_name)

                src_fname = res_root+tag_nob_folder+'top_potsdam_' + str(i1)+'_'+str(i2)+'.png'
                tar_path = tar_root + 'tag_nob/'
                im = Image.open(src_fname)
                l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
                root_name = tar_path + 'PD_' + str(fid).zfill(2) + '_'
                x_image_split(l_corners, patch_size, im, root_name)

                #x_image_split_no_aug(l_corners, patch_size, im, root_name)
                fid += 1


def xf_PD_data_split_test(patch_size, overlap):
    """
    Split the Potsdam dataset with certain overlap and patch size
    :param patch_size: [height, weight] in pixels
    :param overlap: [height, weight] in scale [0,1]
    :return:
    """
    print 'Spliting the val plus dataset...'
    res_root = '../PD/raw/'
    rgb_folder = 'rgb/'
    ir_folder = 'ir/'
    ndsm_folder = 'ndsm/'
    tag_folder = 'gt/'
    tag_nob_folder = 'gt_nob/'

    # deal with the rgb first
    n_tile1 = [2,3,4,5,6,7]
    n_tile2 = [12]
    prefix = 'top_potsdam_'
    sufix = 'RGB.jpg'
    data_path = res_root + rgb_folder + prefix
    tar_root = '../PD/split512/test_A2/'

    overlap_size = [int(patch_size[0] * overlap[0]), int(patch_size[1] * overlap[1])]

    fid = 0
    for i1 in n_tile1:
        for i2 in n_tile2:
            src_fname = data_path + str(i1) + '_' + str(i2) + '_' + sufix
            if os.path.isfile(src_fname):
                tar_path = tar_root + 'rgb/'
                im = Image.open(src_fname)
                l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
                root_name = tar_path + 'PD_' + str(fid).zfill(2) + '_'
                x_image_split_no_aug(l_corners, patch_size, im, root_name)

                src_fname = res_root + ndsm_folder + 'dsm_potsdam_' + str(i1).zfill(2) + '_' + str(i2).zfill(
                    2) + '_normalized_ownapproach.jpg'
                if not os.path.isfile(src_fname):
                    print 'Hell:' + str(i1) + '_' + str(i2)
                    src_fname = res_root + ndsm_folder + 'dsm_potsdam_' + str(i1).zfill(2) + '_' + str(i2).zfill(
                        2) + '_normalized_lastools.jpg'
                tar_path = tar_root + 'ndsm/'
                im = Image.open(src_fname)
                l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
                root_name = tar_path + 'PD_' + str(fid).zfill(2) + '_'
                x_image_split_no_aug(l_corners, patch_size, im, root_name)

                src_fname = res_root + ir_folder + 'top_potsdam_' + str(i1) + '_' + str(i2) + '_RGBIR.png'
                tar_path = tar_root + 'ir/'
                im = Image.open(src_fname)
                l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
                root_name = tar_path + 'PD_' + str(fid).zfill(2) + '_'
                x_image_split_no_aug(l_corners, patch_size, im, root_name)

                src_fname = res_root + tag_folder + 'top_potsdam_' + str(i1) + '_' + str(i2) + '.png'
                tar_path = tar_root + 'tag/'
                im = Image.open(src_fname)
                l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
                root_name = tar_path + 'PD_' + str(fid).zfill(2) + '_'
                x_image_split_no_aug(l_corners, patch_size, im, root_name)

                src_fname = res_root + tag_nob_folder + 'top_potsdam_' + str(i1) + '_' + str(i2) + '.png'
                tar_path = tar_root + 'tag_nob/'
                im = Image.open(src_fname)
                l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
                root_name = tar_path + 'PD_' + str(fid).zfill(2) + '_'
                x_image_split_no_aug(l_corners, patch_size, im, root_name)

                # x_image_split_no_aug(l_corners, patch_size, im, root_name)
                fid += 1


def x_write_to_index(str_pre, fname_ind1):
    '''

    Building the index of feature map and ground truth tags file for
    denseImageData layer of Caffe

    :param str_pre: a string prefix direct before the 'split' folder
    :param fname_ind: index file like files.txt
    :return:
    '''
    f_ind = open(fname_ind1, 'w')
    data_root = '../PD/'
    train_sat = 'split/train/rgb'
    folder_root = data_root + train_sat

    for root, dirs, files in os.walk(folder_root):
        for onefile in files:
            s_write = onefile + '\n'
            f_ind.write(s_write)


def xh_test():
    # define the roots
    patch_size = [256, 256]
    overlap = [0, 0]
    # xf_VH_test_split(patch_size, overlap)
    xf_VH_train_split(patch_size, overlap)


if __name__ == '__main__':
    # x_write_to_index(' ', './train_complete.txt')
    xh_test()

