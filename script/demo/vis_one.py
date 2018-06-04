import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import scipy
import scipy.misc
import os

# -------------------------------------------------------- #

ref_data_path = '../data/VH/raw/'
ref_rgb_path = 'top/'
ref_tag_path = 'gts_no_boarder_png/'

tile_result_root = '../misc/'
wbp_root = 'tmp_wBP/'

# test_set = [0,1,2,3,4,5]  # PD test set
test_set = [11, 15, 28, 30, 34]  # VH test set
patch_size = [256, 256]
labels = [0, 1, 2, 3, 4, 5]
N_class = 6

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--name', help="string, input the model name")
parser.add_argument("-t", "--targ", help="target folder")
parser.add_argument("-i", "--img", help="the img to infer")
args = parser.parse_args()
# -------------------------------------------------------- #


def xf_gen_fake_color_map(cind, s_prefix):
    """

    :param cind:
    :param s_prefix:
    :return:
    """
    # color index as showed in the bench mark
    # d_index = [(0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
    d_index = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
    fname_pre = tile_result_root + s_prefix + str(cind) + '.png'
    im_pre = Image.open(fname_pre)
    pre_arr = np.array(im_pre, np.uint8)
    n_osize = pre_arr.shape
    n_nsize = (n_osize[0], n_osize[1], 3)
    im_trans = np.zeros(n_nsize)
    for ih in xrange(n_osize[0]):
        for iw in xrange(n_osize[1]):
            ic = pre_arr[ih, iw]
            icolor = d_index[ic]
            im_trans[ih, iw, :] = icolor
    return im_trans


def xf_save_fake_color_one(tind, s_prefix, s_tar_prefix):
    """

    :param tind:
    :param s_prefix:
    :return:
    """
    im_color = xf_gen_fake_color_map(tind, s_prefix)
    # specify or modify the folder root and name style below
    fname_save = tile_result_root + 'color_result/' + s_tar_prefix + str(tind) + '.png'
    scipy.misc.toimage(im_color).save(fname_save)
    print 'Image is dumped to' + fname_save + ' ! \n'


def xf_save_fake_color_UAV_set(catgory='fcn'):

    # VH color
    #d_index = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
    # BH UAV color
    d_index = [(204, 0, 0), (102, 102, 102), (51, 204, 0), (0, 51, 0), (153, 102, 0), (102, 0, 204)]

    folder_root = '/media/lh/D/Data/Partion1/result/' + catgory + '/'
    targ_folder_root = '/media/lh/D/Data/Partion1/result/' + catgory + '_color/'

    for root, dirs, files in os.walk(folder_root):
        for onefile in files:
            if onefile.endswith('.png'):
                fname_pre = folder_root + onefile

                im_pre = Image.open(fname_pre)
                pre_arr = np.array(im_pre, np.uint8)
                n_osize = pre_arr.shape
                n_nsize = (n_osize[0], n_osize[1], 3)
                im_trans = np.zeros(n_nsize)
                for ih in xrange(n_osize[0]):
                    for iw in xrange(n_osize[1]):
                        ic = pre_arr[ih, iw]
                        icolor = d_index[ic]
                        im_trans[ih, iw, :] = icolor

                fname_save = targ_folder_root + onefile
                scipy.misc.toimage(im_trans).save(fname_save)
                print 'Image is dumped to' + fname_save + ' ! \n'


def xh_save_fake_color_set():
    """
    Generate the color map from the one big int tile

    s_prefix: to indicate the source file
    s_tar_prefix: to indicate the target file name prefix

    :return:
    """
    #s_prefix = 'RSnet_d128_noPS_ois_09cm_area_'
    s_prefix = 'HSN_noBranch_'
    s_tar_prefix = 'HSN_noBranch_color_'
    for cind in test_set:
        xf_save_fake_color_one(cind, s_prefix, s_tar_prefix)


def xf_gen_fake_color_nerror(cind, gt_root, gt_prefix, pred_root, pred_prefix):
    '''
    Paint the result to fake colors and indicate the wrong tagging with specified color defined by d_error
    :param cind: the big tile index number
    :return:
    : im_trans: a np array with size height x weight x 3, which is easy to output as a png image
    '''
    # color index as showed in the bench mark
    # [Attention] here we replace the cluster from pure red(255,0,0) to an ugly(pretty) purple (112,0,140)
    d_index = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (112, 0, 140)]
    d_error = (255, 0, 0) # pure red reserved for error marking

    # initialising and read in the images to array
    fname_gt = gt_root + gt_prefix + str(cind) + '.png'
    # [M]odify the name here to select the source of the color map
    fname_pre = pred_root + pred_prefix + str(cind) + '.png'

    im_pre = Image.open(fname_pre)
    im_gt = Image.open(fname_gt)
    gt_arr = np.array(im_gt, np.uint8)
    pre_arr = np.array(im_pre, np.uint8)
    n_osize = pre_arr.shape
    n_nsize = (n_osize[0], n_osize[1], 3)
    im_trans = np.zeros(n_nsize)

    for ih in xrange(n_osize[0]):
        for iw in xrange(n_osize[1]):
            ic_pre = pre_arr[ih, iw]
            ic_gt = gt_arr[ih, iw]
            if ic_gt == 255 or ic_gt == ic_pre:
                icolor = d_index[ic_pre]
            else:
                icolor = d_error
            im_trans[ih, iw, :] = icolor
    return im_trans


def xf_save_crop_set():
    """
    This function is used to crop the result wBP image to the original size abandon the padding boarders.
    """
    s_prefix = 'pd_ois_50_'
    s_tar_prefix = 'pd_hsnet_wbp_ois_50_'
    gt_root = '../misc/'
    gt_prefix = 'gt_potsdam_'
    for cind in test_set:
        fname_src = tile_result_root+wbp_root+s_prefix+str(cind)+'.png'
        im_src = Image.open(fname_src)
        fname_gt = gt_root+gt_prefix+str(cind)+'.png'
        im_gt = Image.open(fname_gt)
        gt_size = im_gt.size
        im_tar = im_src.crop((0,0,gt_size[0],gt_size[1]))
        fname_tar = tile_result_root+wbp_root+s_tar_prefix+str(cind)+'.png'
        im_tar.save(fname_tar)



def xf_save_fake_color_nerror_one(tind, gt_root, gt_prefix, pred_root, pred_prefix, tar_root, tar_prefix):
    '''
    generate the tag map in which all the error is marked out by specific color of one big tile
    dump one big tile of image to the disk in tile_result_root with specific name
    :param tind:
    :param s_file:
    :param tar_root:
    :return:
    '''
    im_color = xf_gen_fake_color_nerror(tind, gt_root, gt_prefix, pred_root, pred_prefix)
    # [S]pecify or [M]odify the folder root and name style below
    fname_save = tar_root + tar_prefix + str(tind) + '.png'
    scipy.misc.toimage(im_color).save(fname_save)
    print 'Error demo image is dumped to' + fname_save + ' ! \n'


def xh_save_fake_color_nerror_set():
    gt_root = '../misc/'
    gt_prefix = 'top_gt_09cm_area_'
    pred_root = '../misc/'
    # [M]odify pred_prefix to select source files
    pred_prefix = 'RSnet_d128_noPS_ois_09cm_area_'
    tar_root = '../misc/color_result/'
    # [M]odify tar_prefix to select the output files
    tar_prefix = 'RSnet_d128_noPS_ois_nerror_color_09cm_area_'
    for cind in test_set:
        xf_save_fake_color_nerror_one(cind, gt_root, gt_prefix, pred_root, pred_prefix, tar_root, tar_prefix)
    print 'the error color map is produced and saved to: ' + tar_root + tar_prefix


def xd_check_top2_score(cind):
    '''
    generate a heatmap which shows the difference between the scores top1 and top2
    :param cind:
    :return:
    '''
    # load the array and image
    fname_array = tile_result_root + 'hsnet_ois_75_' + str(cind) + '.npy'
    fname_gt = s_raw_data_root + 'top_gt_09cm_area_' + str(cind) + '.png'
    fname_pre = tile_result_root + 'hsnet_ois_75_' + str(cind) + '.png'
    arr_score = np.load(fname_array)
    im_pre = Image.open(fname_pre)
    im_gt = Image.open(fname_gt)
    arr_gt = np.array(im_gt, np.uint8)
    arr_pre = np.array(im_pre, np.uint8)
    n_osize = arr_pre.shape
    n_nsize = (n_osize[0], n_osize[1])
    im_trans = np.zeros(n_nsize)

    for ih in xrange(n_osize[0]):
        for iw in xrange(n_osize[1]):
            ic_pre = arr_pre[ih, iw]
            arr_cur = arr_score[:, ih, iw]
            inds = arr_cur.argsort()[-2:][::-1]
            iscore = arr_score[inds[0], ih, iw] - arr_score[inds[1], ih, iw]
            im_trans[ih, iw] = iscore
    # for ih in xrange(550, 1000):
    #     print im_trans[ih, 550:1000]

    plt.imshow(im_trans)
    plt.show()

    fname_save = tile_result_root + 'top_mosaic_09cm_area_top2d' + str(cind) + '.npy'
    np.save(fname_save, im_trans)


def xf_vis_set():
    """
    Assemble the whole test_set to big tiles
    :param s_prefix:
    :param data_root: specify where the tmp_result is located
    :return:
    """
    tar_set = ['hsnet_ly', 'hsnet_ly_oi', 'hsnet_v3']
    for cur_set in tar_set:
        print 'assemble for: ' + cur_set
        xf_save_fake_color_UAV_set(cur_set)


def xf_save_fake_color_UAV_set_one():
    assemble_root = args.targ[0:-1] + '_assemble/'
    vis_root = args.targ[0:-1] + '_vis/'
    # VH color
    # d_index = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
    # BH UAV color
    d_index = [(204, 0, 0), (102, 102, 102), (51, 204, 0), (0, 51, 0), (153, 102, 0), (102, 0, 204)]

    folder_root = assemble_root
    targ_folder_root = vis_root

    if not os.path.exists(targ_folder_root):
        os.mkdir(targ_folder_root)

    for root, dirs, files in os.walk(folder_root):
        for onefile in files:
            if onefile.endswith('.png'):
                fname_pre = folder_root + onefile

                im_pre = Image.open(fname_pre)
                pre_arr = np.array(im_pre, np.uint8)
                n_osize = pre_arr.shape
                n_nsize = (n_osize[0], n_osize[1], 3)
                im_trans = np.zeros(n_nsize)
                for ih in xrange(n_osize[0]):
                    for iw in xrange(n_osize[1]):
                        ic = pre_arr[ih, iw]
                        icolor = d_index[ic]
                        im_trans[ih, iw, :] = icolor

                fname_save = targ_folder_root + onefile
                scipy.misc.toimage(im_trans).save(fname_save)
                print 'Image is dumped to' + fname_save + ' ! \n'


if __name__ == '__main__':
    xf_save_fake_color_UAV_set_one()
