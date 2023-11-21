# @Github Repo: https://github.com/hongxiangharry/Stochastic-IQT
# @Citation: Lin, H., Figini, M., D'Arco, F., Ogbole, G., Tanno, R., Blumberg, S. B., ... ,and Alexander, D. C. (2023).
# Low-field magnetic resonance image enhancement via stochastic image quality transfer. Medical Image Analysis, 87, 102807.
# @ Please cite the above paper if you feel this code useful for your research.
#
# MIT License
#
# Copyright (c) 2023 Hongxiang Lin <harryhxlin@gmail.com>. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This part is a derivation of ESPCN from 2D to 3D.
# @Special thanks to Dr. Ryutaro Tanno (Google DeepMind) for assistance on programming.
# @Citation for original ESPCN: Shi, Wenzhe, et al. "Real-time single image and video super-resolution using an efficient
#  sub-pixel convolutional neural network." CVPR 2016.
# @Citation for 3D ESPCN: Tanno, Ryutaro, et al. "Uncertainty modelling in deep learning for safer neuroimage enhancement:
#  Demonstration in diffusion MRI." NeuroImage 225 (2021): 117366.
#
""" ESPCN 3D model. """

import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPool3D
from tensorflow.keras.layers import Permute, Reshape
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

def generate_espcn_model(gen_conf, train_conf) :
    K.set_image_data_format(train_conf['data_format'])  # control channels_first / channels_last

    dataset = train_conf['dataset']
    activation = train_conf['activation']
    dimension = train_conf['dimension']
    num_modalities = gen_conf['dataset_info'][dataset]['modalities']
    expected_output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    loss = train_conf['loss']
    metrics = train_conf['metrics']
    optimizer = train_conf['optimizer']
    shrink_dim = gen_conf['dataset_info'][dataset]['shrink_dim']
    sparse_scale = gen_conf['dataset_info'][dataset]['sparse_scale']
    thickness_factor = gen_conf['dataset_info'][dataset]['downsample_scale']

    downsize_factor = train_conf['downsize_factor']
    # is_bn = train_conf['is_bn']
    is_bn = False # by default

    lr = train_conf['learning_rate']
    decay = train_conf['decay']

    input_shape = (num_modalities, ) + patch_shape
    output_shape = (num_modalities, ) + expected_output_shape

    assert dimension in [2, 3]

    # optimizer
    if optimizer == 'Adam':
        optimizer = Adam(lr=lr, decay=decay)
    elif optimizer == 'SGD':
        optimizer = SGD(lr=lr, nesterov=True)

    model = __generate_espcn_model(
        dimension, num_modalities, input_shape, output_shape, activation, shrink_dim=shrink_dim, thickness_factor=thickness_factor, downsize_factor=downsize_factor, is_bn=is_bn)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

def __generate_espcn_model(
    dimension, num_modalities, input_shape, output_shape, activation, shrink_dim, thickness_factor, downsize_factor=1, is_bn=False):

    upsample_size = [1, 1, 1]
    upsample_size[shrink_dim - 1] = 2

    upsample_times = int(np.log2(thickness_factor)) # should be an integer

    input = Input(shape=input_shape)

    conv1 = get_conv_core(dimension, input, int(50/downsize_factor), kernel_size=(3,3,3), is_bn=is_bn) ## 50

    conv2 = get_conv_core(dimension, conv1, int(100/downsize_factor), kernel_size=(1,1,1), is_bn=is_bn) ## 100

    up = get_conv_core(dimension, conv2, int(thickness_factor), kernel_size=(3,3,3), is_bn=is_bn ) ## 4 or 8

    for i in range(upsample_times):
        up = get_deconv_layer(dimension, up, int(thickness_factor / 2**(i+1)), kernel_size = upsample_size, strides=upsample_size) ## 2x, [1, 1, 2]

    pred = organise_output(up, output_shape, activation)

    return Model(inputs=[input], outputs=[pred])


def get_conv_core(dimension, input, num_filters, kernel_size = None, is_bn = False) :
    x = None
    if kernel_size is None:
        kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)

    if dimension == 2 :
        x = Conv2D(num_filters, kernel_size=kernel_size, padding='valid')(input)
        x = Activation('relu')(x)
        if is_bn == True:
            x = BatchNormalization(axis=1)(x)
    else :
        x = Conv3D(num_filters, kernel_size=kernel_size, padding='valid')(input)
        x = Activation('relu')(x)
        if is_bn == True:
            x = BatchNormalization(axis=1)(x)

    return x

def get_deconv_layer(dimension, input, num_filters, kernel_size = None, strides = None) :
    if strides is None:
        strides = (2, 2) if dimension == 2 else (2, 2, 2)
    if kernel_size is None:
        kernel_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=strides)(input)
    else :
        return Conv3DTranspose(num_filters, kernel_size=kernel_size, strides=strides)(input)

def organise_output(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    # pred = Permute((2, 1))(pred)
    ## no activation for image processing case.
    if activation == 'null':
        return pred
    else:
        return Activation(activation)(pred)