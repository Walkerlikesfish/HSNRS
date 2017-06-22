"""
This serves as the test script for the XBP class

Yu Liu @ ETRO VUB
2017
"""

import numpy as np
import matplotlib.image
from matplotlib import pyplot as plt
import math
from PIL import Image
import scipy
import scipy.misc
import XBP
import os
from multiprocessing.dummy import Pool as ThreadPool
import skimage

# [M]odify here to indicate the where the result is located
tile_result_root = '../misc/'
data_root = tile_result_root + 'tmp_result/'

# [M] the root of the raw input
rgb_root = '/home/yuliu/Documents/Data/VH/data/split/set_test/rgb_jpg/'
dsm_root = '/home/yuliu/Documents/Data/VH/data/split/set_test/ndsm_jpg/'

# [M] the result of the wBP is saved in the result_root folder
result_root = tile_result_root + 'tmp_wBP/'

test_set = [0, 1, 2, 3, 4, 5, 6]
patch_size = [256, 256]
labels = [0, 1, 2, 3, 4, 5]


def xt_test():
    mat_base_beliefs = np.ones(shape=(7, 7, 3))
    mat_base_smooth = np.ones(shape=(3, 3))
    c_bp = XBP.XBP(mat_base_smooth, mat_base_beliefs)
    c_bp.x_pass_msg_four()
    print(c_bp.x_calc_belief())


def xi_calc_diffmat(fname):
    # assuming the fname is the full path
    s_filename = fname[-15:]
    s_rgb = rgb_root + s_filename[:-3] + 'jpeg'
    i_rgb = Image.open(s_rgb, 'r')
    a_rgb = np.array(i_rgb)

    mat_diff = np.zeros(shape=[patch_size[0], patch_size[1], 4])
    # to LEFT 3
    mat_working = a_rgb[:, 1:, :]
    mat_working = np.square(a_rgb[:, :-1, :] - mat_working)
    mat_sum = np.sum(mat_working, axis=2)
    mat_diff[:, 1:, 3] = mat_sum
    # to RIGHT 1
    mat_working = a_rgb[:, :-1, :]
    mat_working = np.square(a_rgb[:,1:,:] - mat_working)
    mat_sum = np.sum(mat_working, axis=2)
    mat_diff[:, :-1, 1] = mat_sum
    # to UP 0
    mat_working = a_rgb[1:, :, :]
    mat_working = np.square(a_rgb[:-1,:,:] - mat_working)
    mat_sum = np.sum(mat_working, axis=2)
    mat_diff[1:, :, 0] = mat_sum
    # to DOWN 2
    mat_working = a_rgb[:-1, :, :]
    mat_working = np.square(a_rgb[1:, :, :] - mat_working)
    mat_sum = np.sum(mat_working, axis=2)
    mat_diff[:-1, :, 2] = mat_sum

    mat_diff += 1

    return mat_diff


def xi_sfunc(a, b, T=0.1, v2=20):
    '''
    The smoothness function
    :param a:
    :param b:
    :param T: choose your temperature
    :param v2:
    :return:
    '''
    if a == b:
        c = 1
    else:
        c = 0
    return v2*np.exp(-(1-c)*(1-c)/T)


def xh_infer_set():
    """
    Infer the whole test set
    using multi-threading like 4
    :return:
    """
    # simple multi-threading
    pool = ThreadPool(6)
    pool.map(xh_test_one, test_set)


def x_infer_ind(cind):
    print 'infering tile:' + str(cind)
    for root, dirs, files in os.walk(data_root):
        for onefile in files:
            fname = data_root + onefile
            if onefile.startswith('VH_'+str(cind)) and onefile.endswith('.png'):
                x_infer_one(fname, num_infer=25)
    print 'tile no.' + str(cind) + 'done!'


def x_infer_one(fname, num_infer=20, verbal=False):
    """
    Infer one patch with specified file name, using wBP

    :param fname: file name
    :param num_infer: max infer iterations
    :param verbal: flag to show the reuslt or not
    :return:
    """
    # load the image of fname
    im_label = Image.open(fname)
    arr_label = np.array(im_label, np.uint8)
    if verbal:
        print 'Before the infer:' + fname + '\n'
        plt.imshow(arr_label)
        plt.show()

    # [D]efine the class number
    num_class = 6
    # [D]efine the interation number
    cnt_infer = num_infer

    # build the smooth matrix
    # TODO: define better smooth matrix maybe depend on the score we input
    mat_smooth = np.ones(
        shape=(num_class, num_class),
        dtype=np.float32
    )
    for ic1 in xrange(num_class):
        for ic2 in xrange(num_class):
            mat_smooth[ic1, ic2] = xi_sfunc(ic1, ic2)

    # build the self_msg matrix
    # load the scores to serve as the self_message
    fname_self_msg = fname[0:-4] + '.npy'
    mat_self = np.load(fname_self_msg)
    mat_self = np.transpose(mat_self, (1, 2, 0))
    check_shape = mat_self.shape
    if check_shape[2] > num_class:
        # check if we are using 0 as no label
        mat_self = mat_self[:, :, 1:num_class+1]

    # calc the top2 difference score
    mat_weight = np.ones(shape=(mat_self.shape[0], mat_self.shape[1]))
    for ih in xrange(mat_self.shape[0]):
        for iw in xrange(mat_self.shape[1]):
            arr_cur = mat_self[ih, iw, :]
            inds = arr_cur.argsort()[-2:][::-1]
            iscore = mat_self[ih, iw, inds[0]] - mat_self[ih, iw, inds[1]]
            mat_weight[ih, iw] = iscore
    mat_weight = 10 * np.exp((1+mat_weight)/10)
    np.save(fname[0:-4]+'_wm.npy', mat_weight)

    if verbal:
        print 'the weight map'
        plt.imshow(mat_weight)
        plt.show()

    # build the color confidence map
    # mat_diff = xi_calc_diffmat(fname)

    # create the XBP
    c_bp = XBP.XBP(mat_smooth, mat_self, mat_weight)

    print 'Finish initialisation!'
    cnt_now = 0
    for i in range(cnt_infer):
        cnt_now += 1
        # print(cnt_now)
        c_bp.x_pass_msg_four()
        print 'Finish ' + str(i) + 'iteration'
    arr_infer = c_bp.x_calc_belief()
    save_fname = result_root + fname[-15:]
    scipy.misc.toimage(arr_infer, cmin=0, cmax=255).save(save_fname)

    if verbal:
        print 'after the infer: \n and save to:' + save_fname
        plt.imshow(arr_infer)
        plt.show()


def xh_assemble_one(cind):
    s_raw_root = '/home/yuliu/Documents/Data/VH/data/' + 'top/'
    im_re = x_assemble_one_f(cind, result_root, s_raw_root)
    fname_save = tile_result_root + 'top_mosaic_09cm_area_predict_wBP' + str(cind) + '.png'
    scipy.misc.toimage(im_re, cmin=0, cmax=255).save(fname_save)


def xh_assemble_testset():
    for cind in test_set:
        xh_assemble_one(cind)


def x_assemble_one_f(ind, s_data_root, s_raw_root, verbal=False):
    '''
    assemble the ind^th tag image
    :param ind:
    :return:
    '''
    fname = s_raw_root + 'top_mosaic_09cm_area' + str(ind) + '.tif'
    im = Image.open(fname)
    n_im_size = im.size
    n_w = n_im_size[0]/patch_size[0] + 1
    n_h = n_im_size[1]/patch_size[1] + 1
    im_re = np.zeros((n_h*patch_size[0], n_w*patch_size[1]))
    im_re_c = np.zeros(n_im_size)
    s_pfile_root = 'VH_' + str(ind).zfill(2) + '_'
    pre_id = 0
    # iterate the height
    for ih in range(0, n_h):
        for iw in range(0, n_w):
            fname = s_data_root + s_pfile_root + str(pre_id).zfill(4) + str(0) + '.png'
            pre_id += 1
            im_pre = Image.open(fname)
            im_pre_np = np.asarray(im_pre)
            il = patch_size[1] * iw
            iu = patch_size[0] * ih
            im_re[iu:iu+patch_size[0], il:il+patch_size[1]] = im_pre_np
    im_re_c = im_re[0:n_im_size[1], 0:n_im_size[0]]
    if verbal:
        plt.imshow(im_re_c)
        plt.show()
    return im_re_c


def xh_test_one(cind):
    print 'infering tile: No.' + str(cind) + 'Be a little more patient!'
    s_filename = '../misc/hsnet_pd_ois_50_' + str(cind) + '.png'
    x_infer_one(fname=s_filename, num_infer=6, verbal=False)


if __name__ == '__main__':
    # test_debug()
    # x_infer_ind(11)
    # xh_test_one()
    # x_infer_one('../misc/tmp_result/VH_11_00000.png', num_infer=50, verbal=True)
    # xh_assemble_testset()
    # xh_infer_set()
    xh_test_one(5)


'''
    if f_load:
        mat_self = np.load("mat_self.npy")
    else:
        print "Building the self_msg matrix..."
        mat_self = np.ones(
            shape=(size_label[0], size_label[1], num_class),
            dtype=np.float32
        )
        for ih in xrange(size_label[0]):
            for iw in xrange(size_label[1]):
                for ic in xrange(num_class):
                    ilabel = arr_label[ih][iw]
                    mat_self[ih, iw, ic] = xi_sfunc(ic, ilabel)
        mat_self[:, :] *= (1 / np.sum(mat_self[:, :], axis=2)[:, :, np.newaxis])
        np.save("mat_self.npy", mat_self)
'''
