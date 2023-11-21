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
""" Utilities that account for the pre-processing of data sets. """

import numpy as np
import nibabel as nib
import os
from nibabel.processing import resample_from_to, smooth_image

def preproc_dataset(gen_conf, train_conf) :
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    if dataset == 'HCP-Wu-Minn-Contrast' :
        print('Downsample HCP data with 1D Gaussian filter.')
        preproc_in_idx = dataset_info['postfix_category']['preproc_in'] #
        in_postfix = dataset_info['postfix'][preproc_in_idx] # raw input data name
        preproc_out_idx = dataset_info['postfix_category']['preproc_out'] #
        out_postfix = dataset_info['postfix'][preproc_out_idx] # processed output data name
        return downsample_HCPWuMinnContrast_dataset(dataset_path, dataset_info, in_postfix, out_postfix)

def interp_input(gen_conf, test_conf, interp_order=3) :
    dataset = test_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    if dataset == 'Nigeria19-Multimodal':
        in_postfix = dataset_info['in_postfix'] # raw input data name
        out_postfix = dataset_info['interp_postfix'] # processed output data name
        return upsample_Nigeria19Multimodal_dataset(dataset_path, dataset_info, in_postfix, out_postfix, is_training = False, interp_order=interp_order)

    if dataset == 'MBB':
        in_postfix = dataset_info['in_postfix'] # raw input data name
        out_postfix = dataset_info['interp_postfix'] # processed output data name
        return upsample_MBB_dataset(dataset_path, dataset_info, in_postfix, out_postfix, is_training = False, interp_order=interp_order)

def upsample_MBB_dataset(dataset_path,
                         dataset_info,
                         in_postfix,
                         out_postfix,
                         is_training = True,
                         interp_order = 3):
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']

    if is_training == True:
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    else:
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    voxel_scale = dataset_info['upsample_scale'] # upsample scale

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       in_postfix)
            out_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx],
                                                        out_postfix)
            data = nib.load(in_filename) # load raw data
            i_affine = np.dot(data.affine, np.diag(tuple(1.0/np.array(voxel_scale)) + (1.0, )))  # affine rescaling
            data = resample_from_to(data, (dimensions, i_affine), order=interp_order) # resize
            nib.save(data, out_filename)

    return True

def upsample_Nigeria19Multimodal_dataset(dataset_path,
                                         dataset_info,
                                         in_postfix,
                                         out_postfix,
                                         is_training = True,
                                         interp_order = 3):
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    prefix = dataset_info['prefix']
    ext = dataset_info['format']

    if is_training == True:
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    else:
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    voxel_scale = dataset_info['upsample_scale'] # upsample scale

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       prefix,
                                                       subject_lib[img_idx][:3],
                                                       modality_categories[mod_idx],
                                                       in_postfix,
                                                       ext)
            out_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx],
                                                        prefix,
                                                        subject_lib[img_idx][:3],
                                                        modality_categories[mod_idx],
                                                        out_postfix,
                                                        ext)

            # revise on 14/03/19
            data = nib.load(in_filename) # load raw data
            i_affine = np.dot(data.affine, np.diag(tuple(1.0/np.array(voxel_scale)) + (1.0, )))  # affine rescaling
            data = resample_from_to(data, (dimensions, i_affine), order=interp_order) # resize
            nib.save(data, out_filename)

    return True

def preproc_input(gen_conf, train_conf, is_training = True, interp_order=3) :
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    if dataset == 'Nigeria19-Multimodal':
        in_postfix = dataset_info['postfix'][2] # raw input data name
        out_postfix = dataset_info['postfix'][0] # processed output data name
        return upsample_HCPWuMinnContrast_dataset(dataset_path, dataset_info, in_postfix, out_postfix, is_training = is_training, interp_order=interp_order)

    if dataset == 'HCP-Wu-Minn-Contrast' :
        in_postfix = dataset_info['postfix'][2] # raw input data name
        out_postfix = dataset_info['postfix'][0] # processed output data name
        return upsample_HCPWuMinnContrast_dataset(dataset_path, dataset_info, in_postfix, out_postfix, is_training = is_training, interp_order=interp_order)

    if dataset == 'IBADAN-k8' :
        in_postfix = dataset_info['postfix'][2] # raw input data name
        out_postfix = dataset_info['postfix'][0] # processed output data name
        return upsample_HCPWuMinnContrast_dataset(dataset_path, dataset_info, in_postfix, out_postfix, is_training = is_training, interp_order=interp_order)

def downsample_HCPWuMinnContrast_dataset(dataset_path,
                                         dataset_info,
                                         in_postfix,
                                         out_postfix,
                                         voxel_scale = None):
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']

    subject_lib = dataset_info['training_subjects'] + dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][0] + dataset_info['num_volumes'][1]
    
    if voxel_scale is None:
        downsample_scale = dataset_info['downsample_scale']
        voxel_scale = [1, 1, downsample_scale] # downsample on an axial direction

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       modality_categories[mod_idx],
                                                       in_postfix)
            out_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx],
                                                        modality_categories[mod_idx],
                                                        out_postfix)
            print('Processing \''+in_filename+'\'')
            data = nib.load(in_filename) # load raw data
            fwhm = np.array(data.header.get_zooms()) * [0, 0, downsample_scale] # FWHM of Gaussian filter
            i_affine = np.dot(data.affine, np.diag(voxel_scale + [1]))  # affine rescaling
            i_shape = np.array(data.shape) // voxel_scale  # downsampled shape of output
            data = smooth_image(data, fwhm) # smoothed by FWHM
            data = resample_from_to(data, (i_shape, i_affine)) # resize
            nib.save(data, out_filename)
            print('Save to \''+out_filename+'\'')
            
    return True

def upsample_HCPWuMinnContrast_dataset(dataset_path,
                                       dataset_info,
                                       in_postfix,
                                       out_postfix,
                                       is_training = True,
                                       interp_order = 3):
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']

    if is_training == True:
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    else:
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    voxel_scale = dataset_info['upsample_scale'] # upsample scale


    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       modality_categories[mod_idx],
                                                       in_postfix)
            out_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx],
                                                        modality_categories[mod_idx],
                                                        out_postfix)

            data = nib.load(in_filename) # load raw data
            i_affine = np.dot(data.affine, np.diag(tuple(1.0/np.array(voxel_scale)) + (1.0, )))  # affine rescaling
            data = resample_from_to(data, (dimensions, i_affine), order=interp_order) # resize
            nib.save(data, out_filename)

    return True
