import argparse
import protoGen as pg

parser = argparse.ArgumentParser()


def x_build_test():
    s_proto = pg.generate_data_layer_train_val()
    base_name = 'res1'
    num_output = 128
    bottom_name = 'conv1'
    s_proto += pg.x_generate_resModule_a(name=base_name, bottom=bottom_name, num_output=num_output)
    return s_proto


def x_build_forward():
    s_proto = pg.generate_data_layer_train_val()
    # conv_1 as starter
    s_starter_name = 'conv1'
    s_proto += pg.x_generate_conv_bn_relu(s_starter_name, 'data', s_starter_name, 1, 3, 128)
    # res1 256x256x128 -> 128x128x128
    s_base_name = 'res1'
    s_last_top = 'conv1'
    s_proto += pg.x_generate_resModule_a(name=s_base_name+'_A', bottom=s_last_top, top=s_base_name+'_a', num_output=128)
    s_proto += pg.x_generate_resModule_a(name=s_base_name+'_branch', bottom=s_last_top,
                                         top=s_base_name+'_branch', num_output=128)
    s_proto += pg.generate_pooling_layer(name='pool1',
                                         bottom=s_base_name+'_a',
                                         top='pool1',
                                         pool_type='MAX', kernel_size=2, stride=2)
    # res2 128x128x128 -> 64x64x128
    s_base_name = 'res2'
    s_last_top = 'pool1'
    s_proto += pg.x_generate_resModule_a(name=s_base_name + '_A', bottom=s_last_top, top=s_base_name + '_a',
                                         num_output=128)
    s_proto += pg.x_generate_resModule_a(name=s_base_name + '_branch', bottom=s_last_top,
                                         top=s_base_name + '_branch', num_output=128)
    s_proto += pg.generate_pooling_layer(name='pool2',
                                         bottom=s_base_name + '_a',
                                         top='pool2',
                                         pool_type='MAX', kernel_size=2, stride=2)
    # res3 64x64x128 -> 32x32x128
    s_base_name = 'res3'
    s_last_top = 'pool2'
    s_proto += pg.x_generate_resModule_a(name=s_base_name + '_A', bottom=s_last_top, top=s_base_name + '_a',
                                         num_output=128)
    s_proto += pg.x_generate_resModule_a(name=s_base_name + '_branch', bottom=s_last_top,
                                         top=s_base_name + '_branch', num_output=128)
    s_proto += pg.generate_pooling_layer(name='pool3',
                                         bottom=s_base_name + '_a',
                                         top='pool3',
                                         pool_type='MAX', kernel_size=2, stride=2)
    # res4 32x32x128 -> 32x32x128
    s_base_name = 'res4'
    s_last_top = 'pool3'
    s_proto += pg.x_generate_resModule_a(name=s_base_name + '_A', bottom=s_last_top, top=s_base_name + '_a',
                                         num_output=128)
    s_proto += pg.x_generate_resModule_a(name=s_base_name + '_branch', bottom=s_last_top,
                                         top=s_base_name + '_branch', num_output=128)

    return s_proto


def x_build_backward():
    s_proto = '''
    '''
    # de_res4
    s_base_name = 'de_res4'
    s_last_top = 'res4_a'
    s_proto += pg.x_generate_resModule_a(name=s_base_name + '_A', bottom=s_last_top, top=s_base_name + '_a',
                                         num_output=128)
    s_proto += pg.generate_eltwise_layer(name=s_base_name+ '_sum', bottom0=s_base_name + '_a', bottom1='res4_branch',
                                         top=s_base_name + '_sum')
    s_proto += pg.generate_deconv_layer(name=s_base_name+'_deconv', bottom=s_base_name+'_sum', top=s_base_name+'_fr',
                             num_output=128, kernal_size=4, stride=2)
    s_proto +=  pg.generate_crop_layer(name=s_base_name+'_crop', bottom=s_base_name+'_fr', bottom_ref='pool2',
                           offset=2, top=s_base_name)

    # de_res3
    s_last_top = s_base_name
    s_base_name = 'de_res3'
    s_proto += pg.x_generate_resModule_a(name=s_base_name + '_A', bottom=s_last_top, top=s_base_name + '_a',
                                         num_output=128)
    s_proto += pg.generate_eltwise_layer(name=s_base_name + '_sum', bottom0=s_base_name + '_a', bottom1='res3_branch',
                                         top=s_base_name + '_sum')
    s_proto += pg.generate_deconv_layer(name=s_base_name + '_deconv', bottom=s_base_name + '_sum',
                                        top=s_base_name + '_fr',
                                        num_output=128, kernal_size=4, stride=2)
    s_proto += pg.generate_crop_layer(name=s_base_name + '_crop', bottom=s_base_name + '_fr', bottom_ref='pool1',
                                      offset=2, top=s_base_name)

    # de_res2
    s_last_top = s_base_name
    s_base_name = 'de_res2'
    s_proto += pg.x_generate_resModule_a(name=s_base_name + '_A', bottom=s_last_top, top=s_base_name + '_a',
                                         num_output=128)
    s_proto += pg.generate_eltwise_layer(name=s_base_name + '_sum', bottom0=s_base_name + '_a', bottom1='res2_branch',
                                         top=s_base_name + '_sum')
    s_proto += pg.generate_deconv_layer(name=s_base_name + '_deconv', bottom=s_base_name + '_sum',
                                        top=s_base_name + '_fr',
                                        num_output=128, kernal_size=4, stride=2)
    s_proto += pg.generate_crop_layer(name=s_base_name + '_crop', bottom=s_base_name + '_fr', bottom_ref='data',
                                      offset=2, top=s_base_name)

    # de_res1
    s_last_top = s_base_name
    s_base_name = 'de_res1'
    s_proto += pg.x_generate_resModule_a(name=s_base_name + '_A', bottom=s_last_top, top=s_base_name + '_a',
                                         num_output=128)
    s_proto += pg.generate_eltwise_layer(name=s_base_name + '_sum', bottom0=s_base_name + '_a', bottom1='res1_branch',
                                         top=s_base_name + '_sum')


    # final_conv
    s_proto += pg.generate_conv_layer(name='final_score', bottom=s_base_name+'_sum', top='score', num_output=6,
                                      kernel_size=3, stride=1)

    return s_proto


def x_build_SNET1_conv_part():
    s_proto = pg.generate_data_layer_train_val()
    # conv_1 as starter
    s_top_now = 'conv1_1'
    s_proto += pg.x_generate_conv_bn_relu(s_top_now, 'data', s_top_now, 1, 3, 64)
    s_top_p = s_top_now
    s_top_now = 'conv1_2'
    s_proto += pg.x_generate_conv_bn_relu(s_top_now, s_top_p, s_top_now, 1, 3, 64)
    # conv1 branch
    # s_proto += pg.x_generate_res_module_b(name='conv1_branch', bottom=s_top_now,
    #                                       top='conv1' + '_branch', num_output=64)
    # pool1
    s_top_p = s_top_now
    s_top_now = 'pool1'
    s_proto += pg.generate_pooling_layer(s_top_now, s_top_p, s_top_now,
                                         'MAX', kernel_size=2, stride=2)

    # conv2 as following
    s_top_p = s_top_now
    s_top_now = 'conv2_1'
    s_proto += pg.x_generate_conv_bn_relu(s_top_now, s_top_p, s_top_now, 1, 3, 128)
    s_top_p = s_top_now
    s_top_now = 'conv2_2'
    s_proto += pg.x_generate_conv_bn_relu(s_top_now, s_top_p, s_top_now, 1, 3, 128)
    # conv2 branch [Turn this on when producing the d128]
    s_proto += pg.x_generate_res_module_b(name='conv2_branch', bottom=s_top_now,
                                          top='conv2' + '_branch', num_output=128)
    # pool2
    s_top_p = s_top_now
    s_top_now = 'pool2'
    s_proto += pg.generate_pooling_layer(s_top_now, s_top_p, s_top_now,
                                         'MAX', kernel_size=2, stride=2)

    return s_proto


def x_build_SNET1_conv_part_2():
    s_proto = '''# conv3 -> conv4
    '''
    s_top_now = 'conv3_1'
    s_proto += pg.x_generate_conv_bn_relu(s_top_now, 'pool2', s_top_now, 1, 3, 256)
    s_top_p = s_top_now
    s_top_now = 'conv3_2'
    s_proto += pg.x_generate_conv_bn_relu(s_top_now, s_top_p, s_top_now, 1, 3, 256)

    s_bottom = s_top_now
    s_top_base = 'conv3_3'
    # conv3_1a
    s_proto += pg.x_generate_conv_bn_relu(name=s_top_base+'_1a', bottom=s_bottom, top=s_top_base+'_1a',
                                       stride=1, ks=1, num_output=64)
    # conv3_1b
    s_proto += pg.x_generate_conv_bn_relu(name=s_top_base + '_1br', bottom=s_bottom, top=s_top_base + '_1br',
                                       stride=1, ks=1, num_output=96)
    s_proto += pg.x_generate_conv_bn_relu(name=s_top_base + '_1b', bottom=s_top_base + '_1br', top=s_top_base + '_1b',
                                       stride=1, ks=3, num_output=128)
    # conv3_1c
    s_proto += pg.x_generate_conv_bn_relu(name=s_top_base + '_1cr', bottom=s_bottom, top=s_top_base + '_1cr',
                                       stride=1, ks=1, num_output=32)
    s_proto += pg.x_generate_conv_bn_relu(name=s_top_base + '_1c', bottom=s_top_base + '_1cr', top=s_top_base + '_1c',
                                       stride=1, ks=5, num_output=32)
    # conv3_1d
    s_proto += pg.x_generate_conv_bn_relu(name=s_top_base + '_1dr', bottom=s_bottom, top=s_top_base + '_1dr',
                                       stride=1, ks=1, num_output=32)
    s_proto += pg.x_generate_conv_bn_relu(name=s_top_base + '_1d', bottom=s_top_base + '_1dr', top=s_top_base + '_1d',
                                       stride=1, ks=7, num_output=32)
    s_proto += pg.x_generate_concat_four(name=s_top_base + '_concat', bottom0=s_top_base+'_1a',
                                         bottom1=s_top_base+'_1b', bottom2=s_top_base+'_1c', bottom3=s_top_base+'_1d',
                                         top=s_top_base+'_1all')

    # pool3
    s_top_p = s_top_base+'_1all'
    s_top_now = 'pool3'
    s_proto += pg.generate_pooling_layer(s_top_now, s_top_p, s_top_now,
                                         'MAX', kernel_size=2, stride=2)

    # conv4 as following
    s_top_p = s_top_now
    s_top_now = 'conv4_1'
    s_proto += pg.x_generate_conv_bn_relu(s_top_now, 'pool3', s_top_now, 1, 3, 512)
    s_top_p = s_top_now
    s_top_now = 'conv4_2'
    s_proto += pg.x_generate_conv_bn_relu(s_top_now, s_top_p, s_top_now, 1, 3, 512)

    s_bottom = s_top_now
    s_top_base = 'conv4_3'
    s_proto += pg.x_generate_conv_bn_relu(name=s_top_base + '_1a', bottom=s_bottom, top=s_top_base + '_1a',
                                       stride=1, ks=1, num_output=64)
    # conv4_1b
    s_proto += pg.x_generate_conv_bn_relu(name=s_top_base + '_1br', bottom=s_bottom, top=s_top_base + '_1br',
                                       stride=1, ks=1, num_output=192)
    s_proto += pg.x_generate_conv_bn_relu(name=s_top_base + '_1b', bottom=s_top_base + '_1br', top=s_top_base + '_1b',
                                       stride=1, ks=3, num_output=256)
    # conv4_1c
    s_proto += pg.x_generate_conv_bn_relu(name=s_top_base + '_1cr', bottom=s_bottom, top=s_top_base + '_1cr',
                                       stride=1, ks=1, num_output=64)
    s_proto += pg.x_generate_conv_bn_relu(name=s_top_base + '_1c', bottom=s_top_base + '_1cr', top=s_top_base + '_1c',
                                       stride=1, ks=5, num_output=128)
    # conv4_1d
    s_proto += pg.x_generate_conv_bn_relu(name=s_top_base + '_1dr', bottom=s_bottom, top=s_top_base + '_1dr',
                                       stride=1, ks=1, num_output=32)
    s_proto += pg.x_generate_conv_bn_relu(name=s_top_base + '_1d', bottom=s_top_base + '_1dr', top=s_top_base + '_1d',
                                       stride=1, ks=7, num_output=64)
    s_proto += pg.x_generate_concat_four(name=s_top_base + '_concat', bottom0=s_top_base + '_1a',
                                         bottom1=s_top_base + '_1b', bottom2=s_top_base + '_1c',
                                         bottom3=s_top_base + '_1d',
                                         top=s_top_base + '_1all')

    # conv5
    s_proto += pg.x_generate_conv_bn_relu('conv5', s_top_base + '_1all', 'conv5', 1, 3, 512)

    return s_proto

def x_build_SNET1_inception_part():
    s_proto = '''
#Inception layers\n'''
    s_bottom = 'pool2'
    s_top_base = 'conv3'
    # conv3_1a
    s_proto += pg.x_generate_conv_relu(name=s_top_base+'_1a', bottom=s_bottom, top=s_top_base+'_1a',
                                       stride=1, ks=1, num_output=64)
    # conv3_1b
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1br', bottom=s_bottom, top=s_top_base + '_1br',
                                       stride=1, ks=1, num_output=96)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1b', bottom=s_top_base + '_1br', top=s_top_base + '_1b',
                                       stride=1, ks=3, num_output=128)
    # conv3_1c
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1cr', bottom=s_bottom, top=s_top_base + '_1cr',
                                       stride=1, ks=1, num_output=32)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1c', bottom=s_top_base + '_1cr', top=s_top_base + '_1c',
                                       stride=1, ks=5, num_output=32)
    # conv3_1d
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1dr', bottom=s_bottom, top=s_top_base + '_1dr',
                                       stride=1, ks=1, num_output=32)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1d', bottom=s_top_base + '_1dr', top=s_top_base + '_1d',
                                       stride=1, ks=7, num_output=32)
    s_proto += pg.x_generate_concat_four(name=s_top_base + '_concat', bottom0=s_top_base+'_1a',
                                         bottom1=s_top_base+'_1b', bottom2=s_top_base+'_1c', bottom3=s_top_base+'_1d',
                                         top=s_top_base+'_1all')

    # conv3_2
    s_bottom = s_top_base + '_1all'
    s_top_base = 'conv3'
    # conv3_2a
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2a', bottom=s_bottom, top=s_top_base + '_2a',
                                       stride=1, ks=1, num_output=64)
    # conv3_2b
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2br', bottom=s_bottom, top=s_top_base + '_2br',
                                       stride=1, ks=1, num_output=128)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2b', bottom=s_top_base + '_2br', top=s_top_base + '_2b',
                                       stride=1, ks=3, num_output=128)
    # conv3_2c
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2cr', bottom=s_bottom, top=s_top_base + '_2cr',
                                       stride=1, ks=1, num_output=64)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2c', bottom=s_top_base + '_2cr', top=s_top_base + '_2c',
                                       stride=1, ks=5, num_output=32)
    # conv3_2d
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2dr', bottom=s_bottom, top=s_top_base + '_2dr',
                                       stride=1, ks=1, num_output= 32)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2d', bottom=s_top_base + '_2dr', top=s_top_base + '_2d',
                                       stride=1, ks=7, num_output=32)
    s_proto += pg.x_generate_concat_four(name=s_top_base + '_concat', bottom0=s_top_base + '_2a',
                                         bottom1=s_top_base + '_2b', bottom2=s_top_base + '_2c',
                                         bottom3=s_top_base + '_2d',
                                         top=s_top_base + '_2all')
    # conv3 branch
    s_proto += pg.x_generate_res_module_b(name='conv3_branch', bottom=s_top_base + '_2all',
                                          top=s_top_base+'_branch', num_output=256)
    # pool3
    s_top_p = s_top_base + '_2all'
    s_top_now = 'pool3'
    s_proto += pg.generate_pooling_layer(s_top_now, s_top_p, s_top_now,
                                         'MAX', kernel_size=2, stride=2)

    # conv4
    s_bottom = 'pool3'
    s_top_base = 'conv4'
    # conv4_1a
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1a', bottom=s_bottom, top=s_top_base + '_1a',
                                       stride=1, ks=1, num_output=64)
    # conv4_1b
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1br', bottom=s_bottom, top=s_top_base + '_1br',
                                       stride=1, ks=1, num_output=192)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1b', bottom=s_top_base + '_1br', top=s_top_base + '_1b',
                                       stride=1, ks=3, num_output=256)
    # conv4_1c
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1cr', bottom=s_bottom, top=s_top_base + '_1cr',
                                       stride=1, ks=1, num_output=64)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1c', bottom=s_top_base + '_1cr', top=s_top_base + '_1c',
                                       stride=1, ks=5, num_output=128)
    # conv4_1d
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1dr', bottom=s_bottom, top=s_top_base + '_1dr',
                                       stride=1, ks=1, num_output=32)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1d', bottom=s_top_base + '_1dr', top=s_top_base + '_1d',
                                       stride=1, ks=7, num_output=64)
    s_proto += pg.x_generate_concat_four(name=s_top_base + '_concat', bottom0=s_top_base + '_1a',
                                         bottom1=s_top_base + '_1b', bottom2=s_top_base + '_1c',
                                         bottom3=s_top_base + '_1d',
                                         top=s_top_base + '_1all')

    # conv4_2
    s_bottom = s_top_base + '_1all'
    s_top_base = 'conv4'
    # conv4_2a
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2a', bottom=s_bottom, top=s_top_base + '_2a',
                                       stride=1, ks=1, num_output=64)
    # conv4_2b
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2br', bottom=s_bottom, top=s_top_base + '_2br',
                                       stride=1, ks=1, num_output=256)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2b', bottom=s_top_base + '_2br', top=s_top_base + '_2b',
                                       stride=1, ks=3, num_output=384)
    # conv4_2c
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2cr', bottom=s_bottom, top=s_top_base + '_2cr',
                                       stride=1, ks=1, num_output=32)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2c', bottom=s_top_base + '_2cr', top=s_top_base + '_2c',
                                       stride=1, ks=5, num_output=32)
    # conv4_2d
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2dr', bottom=s_bottom, top=s_top_base + '_2dr',
                                       stride=1, ks=1, num_output=32)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2d', bottom=s_top_base + '_2dr', top=s_top_base + '_2d',
                                       stride=1, ks=7, num_output=32)
    s_proto += pg.x_generate_concat_four(name=s_top_base + '_concat', bottom0=s_top_base + '_2a',
                                         bottom1=s_top_base + '_2b', bottom2=s_top_base + '_2c',
                                         bottom3=s_top_base + '_2d',
                                         top=s_top_base + '_2all')

    # conv4_3
    s_bottom = s_top_base + '_2all'
    s_top_base = 'conv4'
    # conv4_3a
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_3a', bottom=s_bottom, top=s_top_base + '_3a',
                                       stride=1, ks=1, num_output=64)
    # conv4_3b
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_3br', bottom=s_bottom, top=s_top_base + '_3br',
                                       stride=1, ks=1, num_output=256)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_3b', bottom=s_top_base + '_3br', top=s_top_base + '_3b',
                                       stride=1, ks=3, num_output=384)
    # conv4_3c
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_3cr', bottom=s_bottom, top=s_top_base + '_3cr',
                                       stride=1, ks=1, num_output=32)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_3c', bottom=s_top_base + '_3cr', top=s_top_base + '_3c',
                                       stride=1, ks=5, num_output=32)
    # conv4_3d
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_3dr', bottom=s_bottom, top=s_top_base + '_3dr',
                                       stride=1, ks=1, num_output=32)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_3d', bottom=s_top_base + '_3dr', top=s_top_base + '_3d',
                                       stride=1, ks=7, num_output=32)
    s_proto += pg.x_generate_concat_four(name=s_top_base + '_concat', bottom0=s_top_base + '_3a',
                                         bottom1=s_top_base + '_3b', bottom2=s_top_base + '_3c',
                                         bottom3=s_top_base + '_3d',
                                         top=s_top_base + '_3all')
    return s_proto


def x_build_deconv_part_d64():
    s_proto = '''
# Deconvolution part d64\n'''
    # deconv 4
    s_top_base = 'conv4'
    s_proto += pg.generate_deconv_layer(name='deconv_4', bottom=s_top_base+'_3all', top=s_top_base+'_fr',
                                        num_output=256, kernal_size=4, stride=2)
    s_proto += pg.generate_crop_layer(name=s_top_base + '_crop', bottom=s_top_base+'_fr', bottom_ref='conv3_2all',
                                      offset=2, top=s_top_base+'_de')
    s_proto += pg.x_generate_concat_two(name='deconv_4sum', bottom0=s_top_base+'_de', bottom1='conv3_branch',
                                        top='deconv_4sum')
    # deconv 3
    s_bottom = 'deconv_4sum'
    s_top_base = 'deconv3'
    # deconv3_1a
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1a', bottom=s_bottom, top=s_top_base + '_1a',
                                       stride=1, ks=1, num_output=64)
    # deconv3_1b
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1br', bottom=s_bottom, top=s_top_base + '_1br',
                                       stride=1, ks=1, num_output=256)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1b', bottom=s_top_base + '_1br', top=s_top_base + '_1b',
                                       stride=1, ks=3, num_output=128)
    # deconv3_1c
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1cr', bottom=s_bottom, top=s_top_base + '_1cr',
                                       stride=1, ks=1, num_output=64)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1c', bottom=s_top_base + '_1cr', top=s_top_base + '_1c',
                                       stride=1, ks=5, num_output=32)
    # deconv3_1d
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1dr', bottom=s_bottom, top=s_top_base + '_1dr',
                                       stride=1, ks=1, num_output=64)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_1d', bottom=s_top_base + '_1dr', top=s_top_base + '_1d',
                                       stride=1, ks=7, num_output=32)
    s_proto += pg.x_generate_concat_four(name=s_top_base + '_concat', bottom0=s_top_base + '_1a',
                                         bottom1=s_top_base + '_1b', bottom2=s_top_base + '_1c',
                                         bottom3=s_top_base + '_1d',
                                         top=s_top_base + '_1all')

    # deconv3_2
    s_bottom = s_top_base + '_1all'
    s_top_base = 'deconv3'
    # deconv3_2a
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2a', bottom=s_bottom, top=s_top_base + '_2a',
                                       stride=1, ks=1, num_output=64)
    # deconv3_2b
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2br', bottom=s_bottom, top=s_top_base + '_2br',
                                       stride=1, ks=1, num_output=128)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2b', bottom=s_top_base + '_2br', top=s_top_base + '_2b',
                                       stride=1, ks=3, num_output=128)
    # deconv3_2c
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2cr', bottom=s_bottom, top=s_top_base + '_2cr',
                                       stride=1, ks=1, num_output=64)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2c', bottom=s_top_base + '_2cr', top=s_top_base + '_2c',
                                       stride=1, ks=5, num_output=32)
    # deconv3_2d
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2dr', bottom=s_bottom, top=s_top_base + '_2dr',
                                       stride=1, ks=1, num_output=32)
    s_proto += pg.x_generate_conv_relu(name=s_top_base + '_2d', bottom=s_top_base + '_2dr', top=s_top_base + '_2d',
                                       stride=1, ks=7, num_output=32)
    s_proto += pg.x_generate_concat_four(name=s_top_base + '_concat', bottom0=s_top_base + '_2a',
                                         bottom1=s_top_base + '_2b', bottom2=s_top_base + '_2c',
                                         bottom3=s_top_base + '_2d',
                                         top=s_top_base + '_2all')
    # deconv to score
    # s_proto += pg.generate_deconv_layer(name='score_fr', bottom=s_top_base + '_2all', top='score' + '_fr',
    #                                    num_output=6, kernal_size=8, stride=4)
    # s_proto += pg.generate_crop_layer(name='score', bottom='score_fr', bottom_ref='data',
    #                                     offset=2, top='score')

    return s_proto


def x_build_deconv_part_d128():
    s_proto = '''
# Deconvolution part\n'''
    # deconv 3
    s_top_base = 'deconv3'
    s_proto += pg.generate_deconv_layer(name='deconv_3', bottom=s_top_base + '_2all', top=s_top_base + '_fr',
                                        num_output=128, kernal_size=4, stride=2)
    s_proto += pg.generate_crop_layer(name=s_top_base + '_crop', bottom=s_top_base + '_fr', bottom_ref='conv2_2',
                                      offset=2, top=s_top_base + '_de')
    s_proto += pg.x_generate_concat_two(name='deconv_3sum', bottom0=s_top_base + '_de', bottom1='conv2_branch',
                                        top='deconv3_sum')

    s_proto += pg.x_generate_conv_bn_relu(name='deconv3_1', bottom='deconv3_sum', top='deconv3_1',
                                          stride=1, ks=3, num_output=128)
    s_proto += pg.x_generate_conv_bn_relu(name='deconv3_2', bottom='deconv3_1', top='deconv3_2', stride=1, ks=3,
                                          num_output=128)

    # Finally
    s_proto += pg.generate_deconv_layer(name='score_fr', bottom='deconv3_2', top='score_fr',
                                        num_output=6, kernal_size=4, stride=2)
    s_proto += pg.generate_crop_layer(name='score' + '_crop', bottom='score' + '_fr', bottom_ref='data',
                                      offset=2, top='score')

    return s_proto


def xh_write_res_v1():
    s_to_write = x_build_forward()
    s_to_write += x_build_backward()
    # waiting the 'score'
    # s_to_write += pg.generate_last_scores()
    return s_to_write


def xh_write_snet_v1():
    s_to_write = x_build_SNET1_conv_part()
    s_to_write += x_build_SNET1_inception_part()
    s_to_write += x_build_deconv_part_d64()
    s_to_write += x_build_deconv_part_d128()
    # wating the 'score'
    s_to_write += pg.generate_last_scores()
    return s_to_write


def xh_write_snet_d64():
    '''

    Generate the RSnet d64 version

    :return:
    '''
    s_to_write = x_build_SNET1_conv_part()
    s_to_write += x_build_SNET1_inception_part()
    s_to_write += x_build_deconv_part_d64()
    s_to_write += pg.generate_last_scores()
    return s_to_write


def xh_write_snet_d128():
    '''

    Generate the RSnet d128 version

    :return:
    '''
    s_to_write = x_build_SNET1_conv_part()
    s_to_write += x_build_SNET1_inception_part()
    s_to_write += x_build_deconv_part_d64()
    s_to_write += x_build_deconv_part_d128()

    s_to_write += pg.generate_last_scores()
    return s_to_write


if __name__ == '__main__':
    parser.add_argument("-f", "--file", help="indicate the filename for the prototxt")
    args = parser.parse_args()
    fp = open(args.file, 'w')
    fp.write(xh_write_snet_d128())
    fp.close()