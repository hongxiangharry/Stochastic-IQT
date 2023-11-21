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
""" Utilities for reconstructing images (from patch to whole image). """

import itertools
from utils.patching_utils import pad_both_sides
from utils.patching_utils import determine_output_selector
import numpy as np


def reconstruct_volume_imaging(gen_conf, train_conf, patches, volume_shape = None) :
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dimension = train_conf['dimension'] ## 3
    if volume_shape is None:
        volume_shape = dataset_info['dimensions'] # output image size (260, 311, 260)
    extraction_step = train_conf['extraction_step_test'] # shifting step (16, 16, 2)
    output_shape = train_conf['output_shape_test'] # output patch size (16, 16, 16)
    output_nominal_shape = train_conf['output_shape'] # output nominal shape (32, 32, 32)
    sparse_scale = dataset_info['sparse_scale']  ## [1, 1, 8]

    if dimension == 2 :
        output_shape = (0, ) + output_shape if dimension == 2 else output_shape
        extraction_step = (1, ) + extraction_step if dimension == 2 else extraction_step

    # output selector : 21/3/19
    output_extraction_step = tuple(np.array(extraction_step) * sparse_scale)
    output_selector = determine_output_selector(dimension, output_nominal_shape, output_shape)

    # padding
    output_pad_size = ()
    output_pad_expected_shape = ()
    for dim in range(dimension):
        output_pad_size += (output_shape[dim] // 2,) ## (32, 32, 32)
        output_pad_expected_shape += (volume_shape[dim]+output_shape[dim],) # padding image size (292, 343, 292)

    rec_volume = np.zeros(volume_shape)
    rec_volume = pad_both_sides(dimension, rec_volume, output_pad_size) # padding
    rec_patch_count = np.zeros(volume_shape)
    rec_patch_count = pad_both_sides(dimension, rec_patch_count, output_pad_size) # padding

    output_patch_volume = np.ones(output_shape) * 1.0

    coordinates = generate_indexes(
        dimension, output_shape, output_extraction_step, output_pad_expected_shape)

    if dimension == 2 :
        output_shape = (1, ) + output_shape[1:]

    for count, coord in enumerate(coordinates) :
        selection = [slice(coord[i] - output_shape[i], coord[i]) for i in range(len(coord))] # non-padding
        ## rec_volume[selection] += patches[count] # non-padding
        rec_volume[selection] += patches[output_selector][count]   ## non-padding
        rec_patch_count[selection] += output_patch_volume
    # overlapping reconstruction: average patch
    rec_volume = rec_volume/((rec_patch_count == 0)+rec_patch_count)

    # un-padding: 3D
    rec_volume = rec_volume[output_pad_size[0]:-output_pad_size[0],
                 output_pad_size[1]:-output_pad_size[1],
                 output_pad_size[2]:-output_pad_size[2]]
    return rec_volume

def generate_indexes(dimension, output_shape, extraction_step, expected_shape) :
    '''
    expected_shape = (276, 327, 272) padded from (260, 311, 256)
    extraction_step =  (16, 16, 16)
    output_shape =  (16, 16, 16)

    '''
    # expected_shape: total size
    ndims = len(output_shape)

    poss_shape = [output_shape[i] + extraction_step[i] * ((expected_shape[i] - output_shape[i]) // extraction_step[i]) for i in range(ndims)]

    if dimension == 2 :
        output_shape = (1, ) + output_shape[1:]

    idxs = [range(output_shape[i], poss_shape[i] + 1, extraction_step[i]) for i in range(ndims)]
    
    return itertools.product(*idxs)