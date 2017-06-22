import numpy as np
from PIL import Image
import os
import sys

image_root = '/home/yuliu/Documents/Data/VH/data/split/val/rgb_jpg/'
tag_root = '/home/yuliu/Documents/Data/VH/data/split/val/tag_nb_png/'
result_root = '/home/yuliu/Documents/Data/VH/data/split/result/'

test_root = './val/'
test_gt = 'tag/'
test_pred = 'pre/'
test_file = 'VH_11_00044.png'


def x_make_confusion_mat(y_gt, y_pred, labels, lignore):
    '''
    making the confusion matrix using the ground truth and prediction, the labels should be indicated.

    :param y_gt: ground truth matrix
    :param y_pred: predicted label matrix
    :param labels:
    :param lignore: ignore labels
    :return:
    '''
    nlabels = len(labels)
    y_gt = y_gt.flatten()
    y_pred = y_pred.flatten()
    mat_confusion = np.zeros([nlabels, nlabels])

    for clabel in labels:
        inds = np.where(y_pred == clabel)
        inds = inds[0]
        for ind in inds:
            ilabel = y_gt[ind]
            if ilabel != lignore:
                mat_confusion[ilabel, clabel] = mat_confusion[ilabel, clabel] + 1
            else:
                mat_confusion[clabel, clabel] += 0
    return mat_confusion


def x_calc_prec(mat_confusion, labels):
    mat_prec = np.zeros([mat_confusion.shape[1], 1])
    for label in labels:
        tp = mat_confusion[label, label]
        fp = sum(mat_confusion[:, label])
        if fp>0:
            mat_prec[label] = tp / fp
        else:
            mat_prec[label] = 0
    return mat_prec


def x_calc_recall(mat_confusion, labels):
    mat_recall = np.zeros([mat_confusion.shape[1], 1])
    for label in labels:
        tp = mat_confusion[label, label]
        fn = sum(mat_confusion[label, :])
        if fn > 0:
            mat_recall[label] = tp / fn
        else:
            mat_recall[label] = 0
    return mat_recall


def x_calc_f1score(mat_prec, mat_reall, labels):
    mat_fscore = np.zeros([len(mat_reall), 1])
    for label in labels:
        prec = mat_prec[label]
        recall = mat_reall[label]
        if prec + recall > 0:
            mat_fscore[label] = 2 * prec * recall / (prec + recall)
        else:
            mat_fscore[label] = 0
    return mat_fscore


def x_calc_over_prec(mat_confusion, labels):
    s_tp = 0
    s_fp = 0
    for label in labels:
        tp = mat_confusion[label, label]
        fp = sum(mat_confusion[:, label])
        s_tp += tp
        s_fp += fp
    return s_tp/s_fp


def x_calc_over_recall(mat_confusion, labels):
    s_tp = 0
    s_fn = 0
    for label in labels:
        tp = mat_confusion[label, label]
        fn = sum(mat_confusion[label, :])
        s_tp += tp
        s_fn += fn
    s_recall = s_tp/s_fn
    return s_recall


def x_calc_over_fscore(prec, recall):
    return (2 * prec * recall / (prec + recall))


def x_calc_over_acc(mat_confusion, labels):
    s_tp = 0
    s_all = sum(sum(mat_confusion))
    for label in labels:
        tp = mat_confusion[label, label]
        s_tp += tp
    return s_tp / s_all


def xf_set_test_vh():
    root_path = '../misc/'
    gt_prefix = 'raw_VH_set/top_gt_09cm_area_'
    #gt_prefix = 'raw_VH_set/top_gt_withboarder_09cm_area_'
    pred_prefix = 'HSN_A4_'
    #pred_prefix = 'cvpr_'

    labels = [0, 1, 2, 3, 4, 5]
    test_set = [11, 15, 28, 30, 34]
    nlabels = 6

    mat_confusion_s = np.zeros([nlabels, nlabels])
    for cind in test_set:
        gt_fname = root_path + gt_prefix + str(cind) + '.png'
        pred_fname = root_path + pred_prefix + str(cind) + '.png'

        im_gt = Image.open(gt_fname)
        arr_gt = np.array(im_gt)
        im_pred = Image.open(pred_fname)
        arr_pred = np.array(im_pred)
        mat_confusion = x_make_confusion_mat(arr_gt, arr_pred, labels=labels, lignore=255)
        mat_confusion_s += mat_confusion

    mat_prec = x_calc_prec(mat_confusion_s, labels)
    mat_recall = x_calc_recall(mat_confusion_s, labels)
    mat_fscore = x_calc_f1score(mat_prec, mat_recall, labels)
    s_acc = x_calc_over_acc(mat_confusion_s, labels)

    print 'Fscore:'
    print mat_fscore
    print 'Accuracy:'
    print mat_prec
    print 'Overall accuracy:'
    print s_acc


def xf_set_test_pd():
    print 'test of hsnet+ois+wBP gtn'

    root_path = '../misc/'
    gt_prefix = 'gtn_potsdam_'     # ground truth prefix
    pred_prefix = 'tmp_wBP/pd_hsnet_wbp_ois_50_'       # prediction prefix

    labels = [0, 1, 2, 3, 4, 5, 6]
    clabels = [0, 1, 2, 3, 4, 5]

    test_set = [0, 1, 2, 3, 4, 5]
    nlabels = 6

    mat_confusion_s = np.zeros([nlabels, nlabels])
    for cind in test_set:
        gt_fname = root_path + gt_prefix + str(cind) + '.png'
        pred_fname = root_path + pred_prefix + str(cind) + '.png'

        im_gt = Image.open(gt_fname)
        arr_gt = np.array(im_gt)
        im_pred = Image.open(pred_fname)
        arr_pred = np.array(im_pred)
        arr_pred = arr_pred + 1; # only for wBP result
        mat_confusion = x_make_confusion_mat(arr_gt, arr_pred, labels=labels, lignore=0)

        mat_confusion_c = mat_confusion[1:, 1:]
        mat_confusion_s += mat_confusion_c

        mat_prec = x_calc_prec(mat_confusion, clabels)
        mat_recall = x_calc_recall(mat_confusion, clabels)
        mat_fscore = x_calc_f1score(mat_prec, mat_recall, clabels)
        print mat_fscore

    mat_prec = x_calc_prec(mat_confusion_s, clabels)
    mat_recall = x_calc_recall(mat_confusion_s, clabels)
    mat_fscore = x_calc_f1score(mat_prec, mat_recall, clabels)
    s_acc = x_calc_over_acc(mat_confusion_s, clabels)

    print 'Fscore:'
    print mat_fscore
    print 'Accuracy:'
    print mat_prec
    print 'Overall accuracy:'
    print s_acc


def xh_test():
    test_fname_gt = test_root + test_gt + test_file
    test_fname_pred = test_root + test_pred + test_file
    labels = [0, 1, 2, 3, 4, 5]

    im_gt = Image.open(test_fname_gt)
    arr_gt = np.array(im_gt)
    im_pred = Image.open(test_fname_pred)
    arr_pred = np.array(im_pred)
    print arr_gt[100, 0:255]
    print arr_pred[100, 0:255]

    mat_confusion = x_make_confusion_mat(arr_gt, arr_pred, labels=labels, lignore=255)
    print mat_confusion

    mat_prec = x_calc_prec(mat_confusion, labels)
    mat_recall = x_calc_recall(mat_confusion, labels)
    mat_fscore = x_calc_f1score(mat_prec, mat_recall, labels)
    print mat_fscore


if __name__ == '__main__':
    xf_set_test_vh()