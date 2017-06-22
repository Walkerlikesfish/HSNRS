import numpy as np
from PIL import Image
import scipy.misc

# General information of PD and VH dataset
root_folder_PD = '../misc/color_result/PD_set/'
s_file_pre_PD = ['rgb_', 'ndsm_', 'gt_potsdam_', 'fcn_pd_', 'SegNet_color_pd_', 'fpl_pd_', 'hsnet_pd_raw_',
              'hsnet_pd_ois_50_', 'hsnet_wbp_ois_pd_']
s_file_suf_PD = ['.jpg', '.jpg', '.jpg', '.png', '.png' , '.tif', '.png', '.png', '.png']
ind_list_PD = [0, 1, 2, 3, 4, 5]

root_folder_VH = '../misc/color_result/VH_set/'
s_file_pre_VH = ['rgb_', 'dsm_09cm_matching_area', 'gt_color_withboarder_', 'cvpr_',
		      'SegNet_color_','MAP-top_mosaic_09cm_area', 'hsnet_only_color_',
              'hsnet_ois_75_color_', 'hsnet_wbp_ois_75_color_']
s_file_suf_VH = ['.png', '_normalized.jpg', '.png', '.png', '.png', '_class.tif', '.png', '.png', '.png']
ind_list_VH = [11, 15, 28, 30, 34]

# select the dataset for the crop operation
s_file_pre = s_file_pre_PD
s_file_suf = s_file_suf_PD
root_folder = root_folder_PD

crop_folder = '../misc/color_result/crops/'
crop_prefix = 'crop_'


def crop_set(lnu, wnh, cind, crop_id):
    cnt = 0
    for ii in xrange(9):
        fname = root_folder + s_file_pre[ii] + str(cind) + s_file_suf[ii]
        print 'croping file ' + fname
        im = Image.open(fname)
        imc = im.crop((lnu[0], lnu[1], lnu[0] + wnh[0], lnu[1] + wnh[1]))
        fname_save = crop_folder + crop_prefix + str(crop_id) + '_' + str(cnt) + '.png'
        imc.save(fname_save)
        cnt += 1


def assemble_hori(num_images):
    image_list = []
    for ii in xrange(num_images):
        # onefile = root_folder + crop_folder + crop_prefix + str(tileid) + '_' + str(ii) + '.png'
        onefile = crop_folder + 'crop' + str(ii) + '.jpg'
        image_list.append(onefile)
    print image_list

    images = map(Image.open, image_list)
    widths, heights = zip(*(i.size for i in images))

    per_space = 30
    
    total_width = sum(widths)+per_space*(num_images-1)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height), "white")

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
        x_offset += per_space

    new_im.save('test.jpg')


def assemble_vert(tileid):
    image_list = []
    for ii in xrange(9):
        onefile = crop_folder + crop_prefix + str(tileid) + '_' + str(ii) + '.png'
        image_list.append(onefile)

    images = map(Image.open, image_list)
    widths, heights = zip(*(i.size for i in images))

    per_space = 15
    total_space = ii * per_space
    total_width = max(widths)
    total_height = sum(heights) + total_space

    new_im = Image.new('RGB', (total_width, total_height), "white")

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
        y_offset += per_space
    fname_to_save = crop_folder + 'crop' + str(tileid) + '.jpg'
    new_im.save(fname_to_save)


if __name__ == '__main__':
    crop_id = 3
    lnu = [578, 3659]
    wnh = [750, 450]
    # crop_set(lnu, wnh, 4, crop_id)
    # assemble_vert(crop_id)
    assemble_hori(4)
