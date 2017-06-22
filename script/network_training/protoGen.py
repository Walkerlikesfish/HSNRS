def generate_data_layer_train_val():
    data_layer_str = '''name: "SNet"
layer{
  name: "data_rgb"
  type: "Data"
  top: "data_rgb"
  include {
    phase: TRAIN
  }
  transform_param {
    mean_value: 73.83527
    mean_value: 74.98086
    mean_value: 108.47272
  }
  data_param {
    source: "/home/yuliu/Documents/0RSNet/data/VH/train/rgb"
    batch_size: 4
    backend: LMDB
  }
}
layer {
  name: "data_ndsm"
  type: "Data"
  top: "data_ndsm"
  include {
    phase: TRAIN
  }
  transform_param {
    mean_value: 128.77069
  }
  data_param {
    source: "/home/yuliu/Documents/0RSNet/data/VH/train/ndsm"
    batch_size: 4
    backend: LMDB
  }
}
layer {
  name: "label"
  type: "Data"
  top: "label"
  include {
    phase: TRAIN
  }
  data_param {
    source: "/home/yuliu/Documents/0RSNet/data/VH/train/tag"
    batch_size: 4
    backend: LMDB
  }
}
layer {
  name: "data_rgb"
  type: "Data"
  top: "data_rgb"
  include {
    phase: TEST
  }
  transform_param {
    mean_value: 73.83527
    mean_value: 74.98086
    mean_value: 108.47272
  }
  data_param {
    source: "/home/yuliu/Documents/0RSNet/data/VH/val/rgb"
    batch_size: 4
    backend: LMDB
  }
}
layer {
  name: "data_ndsm"
  type: "Data"
  top: "data_ndsm"
  include {
    phase: TEST
  }
  transform_param {
    mean_value: 128.77069
  }
  data_param {
    source: "/home/yuliu/Documents/0RSNet/data/VH/val/ndsm"
    batch_size: 4
    backend: LMDB
  }
}
layer {
  name: "label"
  type: "Data"
  top: "label"
  include {
    phase: TEST
  }
  data_param {
    source: "/home/yuliu/Documents/0RSNet/data/VH/val/tag"
    batch_size: 4
    backend: LMDB
  }
}
#Concat the dsm
layer {
  name: "concat_data"
  bottom: "data_rgb"
  bottom: "data_ndsm"
  top: "data"
  type: "Concat"
  concat_param {
    axis: 1
  }
}
    \n'''
    return data_layer_str


def generate_data_layer_train_val_b():
    data_layer_str = '''name: "SNet"
layer{
  name: "data_rgb"
  type: "Data"
  top: "data_rgb"
  include {
    phase: TRAIN
  }
  transform_param {
    mean_value: 73.83527
    mean_value: 74.98086
    mean_value: 108.47272
  }
  data_param {
    source: "/home/yuliu/Documents/Data/VH/data/split/set_lmdb/train/rgb"
    batch_size: 4
    backend: LMDB
  }
}
layer {
  name: "data_ndsm"
  type: "Data"
  top: "data_ndsm"
  include {
    phase: TRAIN
  }
  transform_param {
    mean_value: 128.77069
  }
  data_param {
    source: "/home/yuliu/Documents/Data/VH/data/split/set_lmdb/train/ndsm"
    batch_size: 4
    backend: LMDB
  }
}
layer {
  name: "label"
  type: "Data"
  top: "label"
  include {
    phase: TRAIN
  }
  data_param {
    source: "/home/yuliu/Documents/Data/VH/data/split/set_lmdb/train/tag"
    batch_size: 4
    backend: LMDB
  }
}
layer {
  name: "data_rgb"
  type: "Data"
  top: "data_rgb"
  include {
    phase: TEST
  }
  transform_param {
    mean_value: 73.83527
    mean_value: 74.98086
    mean_value: 108.47272
  }
  data_param {
    source: "/home/yuliu/Documents/Data/VH/data/split/set_lmdb/val/rgb"
    batch_size: 4
    backend: LMDB
  }
}
layer {
  name: "data_ndsm"
  type: "Data"
  top: "data_ndsm"
  include {
    phase: TEST
  }
  transform_param {
    mean_value: 128.77069
  }
  data_param {
    source: "/home/yuliu/Documents/Data/VH/data/split/set_lmdb/val/ndsm"
    batch_size: 4
    backend: LMDB
  }
}
layer {
  name: "label"
  type: "Data"
  top: "label"
  include {
    phase: TEST
  }
  data_param {
    source: "/home/yuliu/Documents/Data/VH/data/split/set_lmdb/val/tag"
    batch_size: 4
    backend: LMDB
  }
}
#Concat the dsm
layer {
  name: "concat_data"
  bottom: "data_rgb"
  bottom: "data_ndsm"
  top: "data"
  type: "Concat"
  concat_param {
    axis: 1
  }
}
    \n'''
    return data_layer_str


def generate_conv_layer(name, bottom, top, num_output, kernel_size, stride, filler='msra', std=0.01, lr_mult=1, decay_mult=1, lr_mult2=2, decay_mult2=0):
    pad = (kernel_size - 1) / 2
    conv_layer_str = '''
layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Convolution"
  param { lr_mult: %d decay_mult: %d }
  param { lr_mult: %d decay_mult: %d }
  convolution_param {
    num_output: %d
    kernel_size: %d
    pad: %d
    stride: %d
    weight_filler { type: "%s" std: %.3f }
    bias_filler { type: "constant" value: 0 }
  }
}\n'''%(name, bottom, top, lr_mult, decay_mult, lr_mult2, decay_mult2, num_output, kernel_size, pad, stride, filler, std)
    return conv_layer_str


def generate_conv_layer_no_bias(name, bottom, top, lr_mult, decay_mult, num_output, kernel_size, stride, filler='msra', std=0.01):
    pad = (kernel_size - 1) / 2
    conv_layer_str = '''
layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Convolution"
  param { lr_mult: %d decay_mult: %d }
  convolution_param {
    num_output: %d
    kernel_size: %d
    pad: %d
    stride: %d
    bias_term: false
    weight_filler { type: "%s" std: %.3f }
  }
}\n'''%(name, bottom, top, lr_mult, decay_mult, num_output, kernel_size, pad, stride, filler, std)
    return conv_layer_str


def generate_pooling_layer(name, bottom, top, pool_type, kernel_size, stride):
    pool_layer_str = '''
layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Pooling"
  pooling_param {
    pool: %s
    kernel_size: %d
    stride: %d
  }
}\n'''%(name, bottom, top, pool_type, kernel_size, stride)
    return pool_layer_str


def generate_fc_layer(name, bottom, top, num_output, filler="msra", std=0.01):
    fc_layer_str = '''
layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  param { lr_mult: 1 decay_mult: 1 }
  param { lr_mult: 2 decay_mult: 0 }
  inner_product_param {
     num_output: %d
     weight_filler { type: "%s" std: %.3f }
     bias_filler { type: "constant" value: 0 }
  }
}\n'''%(name, bottom, top, num_output, filler, std)
    return fc_layer_str


def generate_activation_layer(name, bottom, act_type="ReLU"):
    act_layer_str = '''
layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "%s"
}\n'''%(name, bottom, bottom, act_type)
    return act_layer_str


def generate_softmax_layer(name, bottom, top):
    softmax_layer_str = '''
layer {
  name: "%s"
  type: "Softmax"
  bottom: "%s"
  top: "%s"
}\n'''%(name, bottom, top)
    return softmax_layer_str


def generate_softmax_loss(name, bottom0, bottom1, top):
    softmax_loss_str = '''
layer {
  name: "%s"
  type: "SoftmaxWithLoss"
  bottom: "%s"
  bottom: "%s"
  propagate_down: 1
  propagate_down: 0
  top: "%s"
  loss_weight: 1
}\n'''%(name, bottom0, bottom1, top)
    return softmax_loss_str


def generate_smoothl1_loss(name, bottom, top):
    smoothl1_loss_str = '''
layer {
  name: "%s"
  type: "SmoothL1Loss"
  bottom: "%s_pred"
  bottom: "%s_targets"
  bottom: "%s_inside_weights"
  bottom: "%s_outside_weights"
  top: "%s"
  loss_weight: 1
}\n'''%(name, bottom, bottom, bottom, bottom, top)
    return smoothl1_loss_str


def generate_bn_layer_deploy(bn_name, scale_name, bottom):
    bn_layer_str = '''
layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "BatchNorm"
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Scale"
  scale_param {
    bias_term: true
  }
}\n'''%(bn_name, bottom, bottom, scale_name, bottom, bottom)
    return bn_layer_str


def generate_bn_layer(bn_name, scale_name, bottom, use_global_stats=True):
    # use_global_stats: set true in testing, false otherwise.
    if use_global_stats:
        ugs = 'true'
        lr_mult = 0
    else:
        ugs = 'false'
        lr_mult = 1

    bn_layer_str = '''
layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "BatchNorm"
  batch_norm_param {
    use_global_stats: %s
  }
  param { lr_mult: 0 }
  param { lr_mult: 0 }
  param { lr_mult: 0 }
  include {
    phase: TRAIN
  }
}
\n''' % (bn_name, bottom, bottom, 'false')

    bn_layer_str += '''
layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "BatchNorm"
  batch_norm_param {
    use_global_stats: %s
  }
  param { lr_mult: 0 }
  param { lr_mult: 0 }
  param { lr_mult: 0 }
  include {
    phase: TEST
  }
}
\n''' % (bn_name, bottom, bottom, 'true')

    bn_layer_str += '''
layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Scale"
  scale_param {
    bias_term: true
  }
}
''' % (scale_name, bottom, bottom)

    return bn_layer_str


def generate_eltwise_layer(name, bottom0, bottom1, top):
    eltwise_layer_str = '''
layer {
  name: "%s"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
  type: "Eltwise"
}\n'''%(name, bottom0, bottom1, top)
    return eltwise_layer_str


def generate_deconv_layer(name, bottom, top, num_output, kernal_size, stride):
    deconv_str = '''
layer {
  name: "%s"
  type: "Deconvolution"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1.0
  }
  convolution_param {
    num_output: %d
    bias_term: false
    kernel_size: %d
    stride: %d
    weight_filler {
      type: "bilinear"
    }
  }
}\n'''%(name, bottom, top, num_output, kernal_size, stride)
    return deconv_str


def generate_crop_layer(name, bottom, bottom_ref, top, offset):
    crop_str = '''
layer {
  name: "%s"
  type: "Crop"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
  crop_param {
    axis: 2
    offset: %d
    offset: %d
  }
}\n'''%(name, bottom, bottom_ref, top, offset, offset)
    return crop_str


def generate_last_scores():
    last_score = '''
layer {
  name: "loss"
  type: "SoftmaxWithWeightLoss"
  bottom: "score"
  bottom: "label"
  top: "loss"
  loss_param {
    ignore_label: 255
    normalize: true
    weight_by_label_freqs: true
    class_weighting: 3.0
    class_weighting: 2.2
    class_weighting: 3.2
    class_weighting: 2
    class_weighting: 13.0
    class_weighting: 1.52
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "score"
  bottom: "label"
  top: "accuracy"
  top: "per_class_accuracy"
  include {
    phase: TEST
  }
  accuracy_param {
    ignore_label: 255
  }
}
layer {
  name: "accuracy_top_2"
  type: "Accuracy"
  bottom: "score"
  bottom: "label"
  top: "accuracy_top_2"
  include {
    phase: TEST
  }
  accuracy_param {
    top_k: 2
    ignore_label: 255
  }
}\n'''
    return last_score


def x_generate_conv_bn_relu(name, bottom, top, stride, ks, num_output):
    s_cbr = '''
    '''
    s_cbr += x_generate_conv_bn(name, bottom, top, stride, ks, num_output)
    s_cbr += generate_activation_layer(name+'_relu', top)
    return s_cbr


def x_generate_conv_bn(name, bottom, top, stride, ks, num_output, lr_mult=1, lr_mult2=1, decay_mult=1):
    s_cb = '''
    '''
    s_cb += generate_conv_layer(name=name, bottom=bottom, top=top, stride=stride, kernel_size=ks,
                                num_output=num_output)
    s_cb += generate_bn_layer(bn_name=name+'bn', scale_name=name+'scale', bottom=top, use_global_stats=False)
    return s_cb


def x_generate_conv_relu(name, bottom, top, stride, ks, num_output, lr_mult1=1, lr_mult2=1, decay_mult=1):
    s_cr = '''
    '''
    s_cr += generate_conv_layer(name=name, bottom=bottom, top=top, stride=stride, kernel_size=ks,
                                num_output=num_output, lr_mult=lr_mult1, decay_mult=decay_mult, lr_mult2=lr_mult2)
    s_cr += generate_activation_layer(name=name+'_relu', bottom=top)
    return s_cr


def x_generate_concat_four(name, bottom0, bottom1, bottom2, bottom3, top):
    s_concat = '''
layer {
  name: "%s"
  bottom: "%s"
  bottom: "%s"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
  type: "Concat"
  concat_param {
    axis: 1
  }
}
    '''%(name, bottom0, bottom1, bottom2, bottom3, top)
    return s_concat

def x_generate_concat_two(name, bottom0, bottom1, top):
    s_concat = '''
layer {
  name: "%s"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
  type: "Concat"
  concat_param {
    axis: 1
  }
}
    '''%(name, bottom0, bottom1, top)
    return s_concat


def x_generate_resModule_a(name, bottom, top, num_output):
    s_ResMA = '''#
    '''
    # skip
    s_name = name + '_skip'
    s_bottom = bottom
    s_top = name + '_skip'
    s_ResMA += x_generate_conv_bn(s_name, s_bottom, s_top, stride=1, ks=1, num_output=num_output)
    # main road
    s_name = name + '_main'
    s_ResMA += x_generate_conv_bn_relu(name=s_name+'_1', bottom=bottom,
                                       top=s_name+'_1', stride=1, ks=1, num_output=num_output/2)
    s_ResMA += x_generate_conv_bn_relu(name=s_name+'_2', bottom=s_name+'_1',
                                       top=s_name+'_2', stride=1, ks=3, num_output=num_output/2)
    s_ResMA += x_generate_conv_bn(name=s_name+'_3', bottom=s_name+'_2',
                                  top=s_name+'_3', stride=1, ks=1, num_output=num_output)
    # combine skip + main_road
    s_ResMA += generate_eltwise_layer(name=name+'_eltsum', bottom0=s_name+'_3', bottom1=name+'_skip', top=top)
    s_ResMA += generate_activation_layer(name=name+'_relu', bottom=top)

    # top: name_sum
    return s_ResMA


def x_generate_res_module_b(name, bottom, top, num_output):
    s_resb = '''
# Residual module'''
    # skip
    s_resb += generate_conv_layer(name=name+'_skip', bottom=bottom, top=name+'_skip',
                                  num_output=num_output, kernel_size=1, stride=1)
    # main road
    s_resb += generate_conv_layer(name=name+'_1', bottom=bottom, top=name+'_1',
                                  num_output=64, kernel_size=1, stride=1)
    s_resb += generate_activation_layer(name=name+'_r1', bottom=name+'_1')

    s_resb += generate_conv_layer(name=name + '_2', bottom=name+'_1', top=name + '_2',
                                  num_output=128, kernel_size=3, stride=1)
    s_resb += generate_activation_layer(name=name + '_r2', bottom=name + '_2')
    s_resb += generate_conv_layer(name=name + '_3', bottom=name + '_2', top=name + '_3',
                                  num_output=num_output, kernel_size=1, stride=1)
    # eltwise
    s_resb += generate_eltwise_layer(name=name + '_eltsum', bottom0=name+'_3', bottom1=name+'_skip', top=top)
    s_resb += generate_activation_layer(name=name + '_relu', bottom=top)

    return s_resb