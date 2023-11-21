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
""" loading/saving modules:
    1. mean/std; 2. data directory path; 3. training dataloader
"""

import nibabel as nib
import numpy as np
import os
from architectures.arch_creator import generate_model
from numpy.random import shuffle
from dataloader.HCP import DefineTrainValHCPDataloader
from dataloader.HCP import read_dataloader_meanstd as read_dataloader_meanstd_HCP
from dataloader.HCP import save_dataloader_meanstd as save_dataloader_meanstd_HCP

'''
    Module 1: read and save mean/std
'''
def read_dataloader_meanstd(gen_conf, conf, i_dataset=None) :
    dataset = conf['dataset']
    if dataset in ['HCP', 'HCP-Wu-Minn-Contrast-Multimodal', 'HCP-Wu-Minn-Contrast-Multimodal-test',  'HCP-rc', 'HCP-rc-test',
                   'MBB-rc', 'MBB-rc-test']:
        mean, std = read_dataloader_meanstd_HCP(gen_conf, conf, i_dataset=i_dataset)
    return mean, std

def save_dataloader_meanstd(gen_conf, train_conf, mean, std, i_dataset = None) :
    dataset = train_conf['dataset']
    if dataset in ['HCP', 'HCP-Wu-Minn-Contrast-Multimodal', 'HCP-Wu-Minn-Contrast-Multimodal-test', 'HCP-rc', 'HCP-rc-test',
                   'MBB-rc', 'MBB-rc-test']:
        return save_dataloader_meanstd_HCP(gen_conf, train_conf, mean, std, i_dataset=i_dataset)

def read_meanstd(gen_conf, test_conf, case_name = 0) :
    mean_filename = generate_output_filename(
            gen_conf['model_path'],
            test_conf['dataset'],
            case_name,
            test_conf['approach'],
            test_conf['dimension'],
            str(test_conf['patch_shape']),
            str(test_conf['extraction_step'])+'_mean',
            'npz')
    mean = {}
    mean_f = np.load(mean_filename)
    mean['input'] = mean_f['mean_input']
    mean['output'] = mean_f['mean_output']
        
    std_filename = generate_output_filename(
            gen_conf['model_path'],
            test_conf['dataset'],
            case_name,
            test_conf['approach'],
            test_conf['dimension'],
            str(test_conf['patch_shape']),
            str(test_conf['extraction_step'])+'_std',
            'npz')
    std = {}
    std_f = np.load(std_filename)
    std['input'] = std_f['std_input']
    std['output'] = std_f['std_output']
    return mean, std

def save_meanstd(gen_conf, train_conf, mean, std, case_name = 0):
    ## save mean and std
    mean_filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        case_name,
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']) + '_mean',
        'npz')
    std_filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        case_name,
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']) + '_std',
        'npz')
    ## check and make folders
    meanstd_foldername = os.path.dirname(mean_filename)
    if not os.path.isdir(meanstd_foldername):
        os.makedirs(meanstd_foldername)

    if (mean is None) or (std is None):
        mean = {'input': np.array([0.0]), 'output': np.array([0.0])}
        std = {'input': np.array([1.0]), 'output': np.array([1.0])}
    np.savez(mean_filename, mean_input=mean['input'], mean_output=mean['output'])
    np.savez(std_filename, std_input=std['input'], std_output=std['output'])
    return True

def save_random_samples(gen_conf, train_conf, ran_samples, case_name = 1):
    filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        case_name,
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']) + '_random_samples',
        'npz')
    ## check and make folders
    foldername = os.path.dirname(filename)
    if not os.path.isdir(foldername):
        os.makedirs(foldername)

    np.savez(filename, ran_samples=ran_samples)
    return True

def read_msecorr_array(gen_conf, train_conf) :
    filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        'mse_corr',
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'npz')
    data = np.load(filename)
    return data['mse_array'], data['corr_array']

def save_msecorr_array(gen_conf, train_conf, mse_array, corr_array) :
    filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        'mse_corr',
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'npz')
    ## check and make folders
    foldername = os.path.dirname(filename)
    if not os.path.isdir(foldername):
        os.makedirs(foldername)
    np.savez(filename, mse_array=mse_array, corr_array=corr_array)
    return True

def read_model(gen_conf, train_conf, case_name = None) :
    if case_name is None:
        case_name = 'all'

    model = generate_model(gen_conf, train_conf)

    model_filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        case_name,
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'h5')

    model.load_weights(model_filename)

    return model

'''
    Module 2: Read dataset path for simulator and normaliser and test time
'''
def read_data_path(gen_conf, traintest_conf, trainTestFlag = 'train', originProcessFlag = 'origin', i_dataset = None, indices = None):
    '''
    :param gen_conf: require 'path' to be a 4-item list
        0: processed image after simulation and normalisation
        1: original image
        2: patch lib on scratch
        3: patch lib on local
    '''
    dataset = traintest_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    if dataset in ['Nigeria19-rc', 'Nigeria17-rc']:
        return read_Nigeria19_rc_path(dataset_path, dataset_info, trainTestFlag, originProcessFlag, i_dataset, indices)
    if dataset in ['HCP', 'HCP-Wu-Minn-Contrast-Multimodal', 'HCP-Wu-Minn-Contrast-Multimodal-test', 'HCP-rc', 'HCP-rc-test',
                   'MBB-rc', 'MBB-rc-test']:
        return read_HCPWuMinnContrastMultimodal_path(gen_conf, traintest_conf, trainTestFlag, originProcessFlag, i_dataset, indices)
    if dataset in ['HCP-LRTV', 'HCP-cubic', 'HCP-GT', 'HCP-input']:
        return read_HCP_single_contrast_path(gen_conf, traintest_conf, trainTestFlag, originProcessFlag, i_dataset, indices)

def read_HCP_single_contrast_path(gen_conf, traintest_conf,
                    trainTestFlag = 'train',
                    originProcessFlag = 'origin',
                    i_dataset = None,
                    indices = None):
    # data
    dataset = traintest_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    modalities = dataset_info['modalities']

    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']

    # trainTestFlag
    if trainTestFlag in ['test', 'eval'] :
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    else :
        raise ValueError("trainTestFlag should be declared as 'train'/'eval'/'test'..")

    # originProcessFlag
    if originProcessFlag == 'origin':
        path = dataset_info['path'][1]

    elif originProcessFlag == 'process':
        path = dataset_info['path'][0]
    else:
        raise ValueError("originProcessFlag should be specified as 'origin'/'process'")

    in_postfix = dataset_info['in_postfix']
    out_postfix = dataset_info['out_postfix']

    # initialise in/out_paths
    in_paths = []
    out_paths = []

    if indices is None:
        img_indices = range(num_volumes) # num of train/test subjects
        mod_indices = range(modalities)
    else:
        img_indices = [indices[0]]
        mod_indices = [indices[1]]

    for img_idx in img_indices:
        for mod_idx in mod_indices:
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       modality_categories[mod_idx],
                                                       in_postfix)
            in_paths.append(in_filename)
            if trainTestFlag != 'test':
                out_filename = os.path.join(dataset_path,
                                            path,
                                            pattern).format(subject_lib[img_idx],
                                                            modality_categories[mod_idx],
                                                            out_postfix)
                out_paths.append(out_filename)
            else:
                out_paths = None

    return in_paths, out_paths

def read_HCPWuMinnContrastMultimodal_path(gen_conf, traintest_conf,
                                        trainTestFlag = 'train',
                                        originProcessFlag = 'origin',
                                        i_dataset = None,
                                        indices = None):
    # data
    dataset = traintest_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    modalities = dataset_info['modalities']

    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']

    # trainTestFlag
    if trainTestFlag == 'train' :
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
        if 'num_train_sbj' in traintest_conf.keys():
            num_volumes = traintest_conf['num_train_sbj']
    elif trainTestFlag in  ['test', 'eval'] :
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    else :
        raise ValueError("trainTestFlag should be declared as 'train'/'eval'/'test'..")

    # originProcessFlag
    if originProcessFlag == 'origin':
        path = dataset_info['path'][1]
        in_postfix = ''
        out_postfix = ''
    elif originProcessFlag == 'process':
        path = dataset_info['path'][0]
        if i_dataset is not None:
            in_postfix = np.array(dataset_info['in_postfix'])[i_dataset] # select SNR, in_postfix: train model x test subject, same as i_dataset
        else:
            in_postfixes = np.array(dataset_info['in_postfix'])
            in_postfix = in_postfixes[(0,) * in_postfixes.ndim]
        out_postfix = dataset_info['out_postfix']
    else:
        raise ValueError("originProcessFlag should be specified as 'origin'/'process'")

    # initialise in/out_paths
    in_paths = []
    out_paths = []

    if indices is None:
        img_indices = range(num_volumes) # num of train/test subjects
        mod_indices = range(modalities)
    else:
        img_indices = [indices[0]]
        mod_indices = [indices[1]]

    for img_idx in img_indices:
        for mod_idx in mod_indices:
            ## for random-contrast case ...
            if not isinstance(in_postfix, str):
                if in_postfix.ndim == 1:
                    in_postfix_sub = in_postfix[img_idx]
                else:
                    in_postfix_sub = in_postfix
            else:
                in_postfix_sub = in_postfix
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       modality_categories[mod_idx],
                                                       in_postfix_sub)
            in_paths.append(in_filename)
            if trainTestFlag != 'test':
                out_filename = os.path.join(dataset_path,
                                            path,
                                            pattern).format(subject_lib[img_idx],
                                                            modality_categories[mod_idx],
                                                            out_postfix)
                out_paths.append(out_filename)
            else:
                out_paths = None

    return in_paths, out_paths

def read_Nigeria19_rc_path(dataset_path, dataset_info, trainTestFlag = 'test',
                            originProcessFlag = 'origin', i_dataset = None, indices = None):
    modalities = dataset_info['modalities']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    prefix = dataset_info['prefix']
    out_postfix = dataset_info['out_postfix']
    ext = dataset_info['format'] # extension format
    if trainTestFlag == 'train' :
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    elif trainTestFlag == 'test' :
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    else :
        raise ValueError("trainTestFlag should be declared as 'train'/'test'/'evaluation'")

    # originProcessFlag
    if originProcessFlag == 'origin':
        path = dataset_info['path'][1]
        in_postfix = dataset_info['in_postfix'][1]
    elif originProcessFlag == 'process':
        path = dataset_info['path'][0]
        in_postfix = dataset_info['in_postfix'][0]
    else:
        raise ValueError("originProcessFlag should be specified as 'origin'/'process'")

    # initialise in/out_paths
    in_paths = []
    out_paths = []

    if indices is None:
        img_indices = range(num_volumes)
        mod_indices = range(modalities)
    else:
        img_indices = [indices[0]]
        mod_indices = [indices[1]]

    for img_idx in img_indices:
        for mod_idx in mod_indices:
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       prefix,
                                                       subject_lib[img_idx][:3],
                                                       modality_categories[mod_idx],
                                                       in_postfix,
                                                       ext)
            in_paths.append(in_filename)
            if trainTestFlag != 'test':
                out_filename = os.path.join(dataset_path,
                                            path,
                                            pattern).format(subject_lib[img_idx],
                                                            prefix,
                                                            subject_lib[img_idx][:3],
                                                            modality_categories[mod_idx],
                                                            out_postfix,
                                                            ext)
                out_paths.append(out_filename)


    return in_paths, out_paths

'''
    Module 3: Read dataset from path or dataloader
'''

def DefineTrainValDataLoader(gen_conf, train_conf, i_dataset = None):
    dataset = train_conf['dataset']
    if dataset in ['HCP', 'HCP-Wu-Minn-Contrast-Multimodal', 'HCP-Wu-Minn-Contrast-Multimodal-test', 'HCP-rc', 'HCP-rc-test', 'MBB-rc', 'MBB-rc-test']:
        train_generator, val_generator = DefineTrainValHCPDataloader(gen_conf, train_conf, i_dataset=i_dataset)
    return train_generator, val_generator

def read_masks(gen_conf, train_conf, trainTestFlag = 'train'):
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    if dataset in ['HCP', 'HCP-Wu-Minn-Contrast-Multimodal', 'HCP-Wu-Minn-Contrast-Multimodal-test',  'HCP-rc', 'HCP-rc-test'] :
        return read_HCPWuMinnContrastMultimodal_masks(dataset_path, dataset_info, trainTestFlag)
    if dataset == ['MBB', 'MBB-rc', 'MBB-rc-test']:
        return read_MBB_masks(dataset_path, dataset_info, trainTestFlag)

def read_HCPWuMinnContrastMultimodal_masks(dataset_path, dataset_info, trainTestFlag = 'train'):
    dimensions = dataset_info['dimensions']
    sparse_scale = dataset_info['sparse_scale']
    modalities = dataset_info['modalities']
    path = dataset_info['path'][0]
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']

    if trainTestFlag == 'train':
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]

    elif trainTestFlag == 'test':
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    else:
        raise ValueError("trainTestFlag should be declared as 'train'/'test'/'evaluation'")

    mask_postfix = dataset_info['mask_postfix']
    data = np.zeros((num_volumes, modalities) + dimensions, dtype=np.bool)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            filename = os.path.join(dataset_path,
                                    path,
                                    pattern).format(subject_lib[img_idx],
                                                    modality_categories[mod_idx],
                                                    mask_postfix)
            data[img_idx, mod_idx] = np.array(read_volume(filename) != 0, dtype=np.bool)

    return data

def read_MBB_masks(dataset_path, dataset_info, trainTestFlag='train'):
    dimensions = dataset_info['dimensions']
    sparse_scale = dataset_info['sparse_scale']
    modalities = dataset_info['modalities']
    path = dataset_info['path'][0]
    pattern = dataset_info['general_pattern']

    if trainTestFlag == 'train':
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    elif trainTestFlag == 'test':
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    else:
        raise ValueError("trainTestFlag should be declared as 'train'/'test'/'evaluation'")

    mask_postfix = dataset_info['mask_postfix']
    data = np.zeros((num_volumes, modalities) + dimensions, dtype=np.bool)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            filename = os.path.join(dataset_path,
                                    path,
                                    pattern).format(subject_lib[img_idx],
                                                    mask_postfix)
            data[img_idx, mod_idx] = np.array(read_volume(filename) != 0, dtype=np.bool)

    return data

def read_dataset(gen_conf, train_conf, trainTestFlag = 'train', i_dataset=None) :

    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    if dataset == 'IBADAN-k8':
        return read_IBADAN_data(dataset_path, dataset_info, trainTestFlag)
    if dataset == 'HBN':
        return read_HBN_dataset(dataset_path, dataset_info, trainTestFlag)
    if dataset == 'HCP-Wu-Minn-Contrast' :
        return read_HCPWuMinnContrast_dataset(dataset_path, dataset_info, trainTestFlag)
    if dataset == 'iSeg2017' :
        return read_iSeg2017_dataset(dataset_path, dataset_info)
    if dataset == 'IBSR18' :
        return read_IBSR18_dataset(dataset_path, dataset_info)
    if dataset == 'MICCAI2012' :
        return read_MICCAI2012_dataset(dataset_path, dataset_info)
    if dataset == 'HCP-Wu-Minn-Contrast-Augmentation' :
        return read_HCPWuMinnContrastAugmentation_dataset(dataset_path, dataset_info, trainTestFlag)
    if dataset in ['HCP', 'HCP-Wu-Minn-Contrast-Multimodal', 'HCP-Wu-Minn-Contrast-Multimodal-test',  'HCP-rc', 'HCP-rc-test'] :
        return read_HCPWuMinnContrastMultimodal_dataset(gen_conf, train_conf, trainTestFlag, i_dataset=i_dataset)
    if dataset == 'Nigeria19-Multimodal' :
        return read_Nigeria19Multimodal_dataset(dataset_path, dataset_info, trainTestFlag)
    if dataset == 'Juntendo-Volunteer':
        return read_JH_Volunteer_dataset(dataset_path, dataset_info, trainTestFlag)
    if dataset == ['MBB', 'MBB-rc', 'MBB-rc-test']:
        return read_MBB_dataset(dataset_path, dataset_info, trainTestFlag)

def read_MBB_dataset(dataset_path,
                     dataset_info,
                     trainTestFlag='train'):
    dimensions = dataset_info['dimensions']
    sparse_scale = dataset_info['sparse_scale']
    input_dimension = tuple(np.array(dimensions) // sparse_scale)
    modalities = dataset_info['modalities']
    path = dataset_info['path'][0]
    pattern = dataset_info['general_pattern']
    in_postfix = dataset_info['in_postfix']
    out_postfix = dataset_info['out_postfix']
    if trainTestFlag == 'train':
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    elif trainTestFlag == 'test':
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    else:
        raise ValueError("trainTestFlag should be declared as 'train'/'test'/'evaluation'")

    in_data = np.zeros((num_volumes, modalities) + input_dimension)
    out_data = np.zeros((num_volumes, modalities) + dimensions)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       in_postfix)
            in_data[img_idx, mod_idx] = read_volume(in_filename)
            out_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx],
                                                        out_postfix)
            out_data[img_idx, mod_idx] = read_volume(out_filename)

    return in_data, out_data

def read_JH_Volunteer_dataset(dataset_path,
                              dataset_info,
                              trainTestFlag='train'):
    dimensions = dataset_info['dimensions']
    sparse_scale = dataset_info['sparse_scale']
    input_dimension = tuple(np.array(dimensions) // sparse_scale)
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    in_postfixes = dataset_info['in_postfixes']
    out_postfixes = dataset_info['out_postfixes']
    in_subfolder = dataset_info['subfolders'][0]
    out_subfolder = dataset_info['subfolders'][1]
    if trainTestFlag == 'train':
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    elif trainTestFlag == 'test':
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    else:
        raise ValueError("trainTestFlag should be declared as 'train'/'test'/'evaluation'")

    in_data = np.zeros((num_volumes, modalities) + input_dimension)
    out_data = np.zeros((num_volumes, modalities) + dimensions)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(in_subfolder,
                                                       subject_lib[img_idx],
                                                       in_postfixes[mod_idx])
            in_data[img_idx, mod_idx] = read_volume(in_filename)
            out_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(out_subfolder,
                                                        subject_lib[img_idx],
                                                        out_postfixes[mod_idx])
            out_data[img_idx, mod_idx] = read_volume(out_filename)

    return in_data, out_data

def read_Nigeria19Multimodal_dataset(dataset_path,
                                     dataset_info,
                                     trainTestFlag = 'test'):
    dimensions = dataset_info['dimensions']
    sparse_scale = dataset_info['sparse_scale']
    input_dimension = tuple(np.array(dimensions) // sparse_scale)
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    prefix = dataset_info['prefix']
    in_postfix = dataset_info['in_postfix']
    out_postfix = dataset_info['out_postfix']
    ext = dataset_info['format'] # extension format
    if trainTestFlag == 'train' :
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    elif trainTestFlag == 'test' :
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    else :
        raise ValueError("trainTestFlag should be declared as 'train'/'test'/'evaluation'")

    in_data = np.zeros((num_volumes, modalities) + input_dimension)
    if trainTestFlag != 'test':
        out_data = np.zeros((num_volumes, modalities) + dimensions)

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
            in_data[img_idx, mod_idx] = read_volume(in_filename)
            if trainTestFlag != 'test':
                out_filename = os.path.join(dataset_path,
                                            path,
                                            pattern).format(subject_lib[img_idx],
                                                            prefix,
                                                            subject_lib[img_idx][:3],
                                                            modality_categories[mod_idx],
                                                            out_postfix,
                                                            ext)
                out_data[img_idx, mod_idx] = read_volume(out_filename)
            else:
                out_data = None

    return in_data, out_data

def read_HCPWuMinnContrastMultimodal_dataset(gen_conf,
                                             train_conf,
                                             trainTestFlag = 'train',
                                             i_dataset = None):
    '''
    rc:  dataset_info['in_postfix'].shape = (n_models, n_test_subjects)

    '''
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    dimensions = dataset_info['dimensions']
    sparse_scale = dataset_info['sparse_scale']
    input_dimension = tuple(np.array(dimensions)//sparse_scale)
    modalities = dataset_info['modalities']
    path = dataset_info['path'][0]
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    # in_postfix = dataset_info['postfix'][0]
    # out_postfix = dataset_info['postfix'][1]
    if i_dataset is not None:
        in_postfix = np.array(dataset_info['in_postfix'])[i_dataset]
    else:
        in_postfixes = np.array(dataset_info['in_postfix'])
        in_postfix = in_postfixes[ (0,)*in_postfixes.ndim]
    out_postfix = dataset_info['out_postfix']
    if trainTestFlag == 'train' :
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
        if 'num_train_sbj' in train_conf:
            num_volumes = train_conf['num_train_sbj']
    elif trainTestFlag == 'test' :
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    else :
        raise ValueError("trainTestFlag should be declared as 'train'/'test'/'evaluation'")

    in_data = np.zeros((num_volumes, modalities) + input_dimension)
    out_data = np.zeros((num_volumes, modalities) + dimensions)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            ## for random-contrast case ...
            if in_postfix.ndim == 1:
                in_postfix_sub = in_postfix[img_idx]
            else:
                in_postfix_sub = in_postfix

            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       modality_categories[mod_idx],
                                                       in_postfix_sub)
            in_data[img_idx, mod_idx] = read_volume(in_filename)
            out_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx],
                                                        modality_categories[mod_idx],
                                                        out_postfix)
            out_data[img_idx, mod_idx] = read_volume(out_filename)

    return in_data, out_data


'''
    todo: 
    description: randomly select 'ran_sub_samples' numbers of samples/patches
    in_data: 6 dim
    out_data: 5 dim
'''
def read_HCPWuMinnContrastAugmentation_dataset(dataset_path,
                                               dataset_info,
                                               trainTestFlag = 'train'):
    dimensions = dataset_info['dimensions']
    sparse_scale = dataset_info['sparse_scale']
    input_dimension = tuple(np.array(dimensions) // sparse_scale)
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    samples = dataset_info['num_samples'][0]
    out_postfix = dataset_info['out_postfix']
    if trainTestFlag == 'train' :
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
        in_postfix = dataset_info['training_in_postfix']
        ran_sub_samples = dataset_info['num_samples'][1]
    elif trainTestFlag == 'test' :
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
        in_postfix = dataset_info['test_in_postfix']
        ran_sub_samples = dataset_info['num_samples'][2] ## one
    else :
        raise ValueError("trainTestFlag should be declared as 'train'/'test'/'evaluation'")

    in_data = np.zeros((num_volumes, modalities, ran_sub_samples) + input_dimension)
    out_data = np.zeros((num_volumes, modalities) + dimensions)

    ran_sub_samples_array_ = []

    # need a way to save
    if trainTestFlag == 'train':
        for img_idx in range(num_volumes):
            for mod_idx in range(modalities):
                ## subsampling on training and test sets
                ran_sub_samples_array = np.arange(samples)
                shuffle(ran_sub_samples_array)
                ran_sub_samples_array = ran_sub_samples_array[:ran_sub_samples]
                ran_sub_samples_array_.append(ran_sub_samples_array) ## unused code !!!
                for smpl_idx, smpl in enumerate(in_postfix[img_idx][mod_idx][idx] for idx in ran_sub_samples_array):
                    ## imput image, random
                    in_filename = os.path.join(dataset_path,
                                                path,
                                                pattern).format(subject_lib[img_idx],
                                                                modality_categories[mod_idx],
                                                                smpl)
                    in_data[img_idx, mod_idx, smpl_idx] = read_volume(in_filename)
                ## output image
                out_filename = os.path.join(dataset_path,
                                            path,
                                            pattern).format(subject_lib[img_idx],
                                                            modality_categories[mod_idx],
                                                            out_postfix)
                out_data[img_idx, mod_idx] = read_volume(out_filename)
    elif trainTestFlag == 'test':
        for img_idx in range(num_volumes):
            for mod_idx in range(modalities):
                for smpl_idx, smpl in enumerate(in_postfix[img_idx][mod_idx][idx] for idx in range(ran_sub_samples)):
                    ## input image, random
                    in_filename = os.path.join(dataset_path,
                                               path,
                                               pattern).format(subject_lib[img_idx],
                                                               modality_categories[mod_idx],
                                                               smpl)
                    in_data[img_idx, mod_idx, smpl_idx] = read_volume(in_filename)
                    ## output image
                    out_filename = os.path.join(dataset_path,
                                                path,
                                                pattern).format(subject_lib[img_idx],
                                                                modality_categories[mod_idx],
                                                                out_postfix)
                    out_data[img_idx, mod_idx] = read_volume(out_filename)

    return in_data, out_data

def read_IBADAN_data(dataset_path,
                     dataset_info,
                     trainTestFlag = 'test'):
    dimensions = dataset_info['dimensions']
    sparse_scale = dataset_info['sparse_scale']
    input_dimension = tuple(np.array(dimensions) // sparse_scale)
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    in_postfix = dataset_info['postfix'][0]
    out_postfix = dataset_info['postfix'][1]
    if trainTestFlag == 'train' :
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    elif trainTestFlag == 'test' :
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    else :
        raise ValueError("trainTestFlag should be declared as 'train'/'test'/'evaluation'")

    in_data = np.zeros((num_volumes, modalities) + input_dimension)
    out_data = np.zeros((num_volumes, modalities) + dimensions)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            in_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx],
                                                        modality_categories[mod_idx],
                                                        in_postfix)
            in_data[img_idx, mod_idx] = read_volume(in_filename)
            if trainTestFlag != 'test' :
                out_filename = os.path.join(dataset_path,
                                            path,
                                            pattern).format(subject_lib[img_idx],
                                                            modality_categories[mod_idx],
                                                            out_postfix)
                out_data[img_idx, mod_idx] = read_volume(out_filename)
            else:
                out_data = None
    return in_data, out_data

def read_HBN_dataset(dataset_path,
                     dataset_info,
                     trainTestFlag = 'train'):
    dimensions = dataset_info['dimensions']
    sparse_scale = dataset_info['sparse_scale']
    input_dimension = tuple(np.array(dimensions)//sparse_scale)
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    in_postfix = dataset_info['postfix'][0]
    out_postfix = dataset_info['postfix'][1]
    if trainTestFlag == 'train' :
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    elif trainTestFlag == 'test' :
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    else :
        raise ValueError("trainTestFlag should be declared as 'train'/'test'/'evaluation'")

    in_data = np.zeros((num_volumes, modalities) + input_dimension)
    out_data = np.zeros((num_volumes, modalities) + dimensions)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            in_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx], subject_lib[img_idx],
                                                        modality_categories[mod_idx],
                                                        in_postfix)
            in_data[img_idx, mod_idx] = read_volume(in_filename)
            out_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx], subject_lib[img_idx],
                                                        modality_categories[mod_idx],
                                                        out_postfix)
            out_data[img_idx, mod_idx] = read_volume(out_filename)

    return in_data, out_data

def read_HCPWuMinnContrast_dataset(dataset_path,
                                   dataset_info,
                                   trainTestFlag = 'train'):
    dimensions = dataset_info['dimensions']
    sparse_scale = dataset_info['sparse_scale']
    input_dimension = tuple(np.array(dimensions) // sparse_scale)
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    # in_postfix = dataset_info['postfix'][0]
    # out_postfix = dataset_info['postfix'][1]
    in_postfix = dataset_info['in_postfix']
    out_postfix = dataset_info['out_postfix']
    if trainTestFlag == 'train' :
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    elif trainTestFlag == 'test' :
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    else :
        raise ValueError("trainTestFlag should be declared as 'train'/'test'/'evaluation'")

    in_data = np.zeros((num_volumes, modalities) + input_dimension)
    out_data = np.zeros((num_volumes, modalities) + dimensions)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       modality_categories[mod_idx],
                                                       in_postfix)
            in_data[img_idx, mod_idx] = read_volume(in_filename)
            out_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx],
                                                        modality_categories[mod_idx],
                                                        out_postfix)
            out_data[img_idx, mod_idx] = read_volume(out_filename)

    return in_data, out_data

def read_iSeg2017_dataset(dataset_path, dataset_info) :
    num_volumes = dataset_info['num_volumes']
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    inputs = dataset_info['inputs']

    image_data = np.zeros((num_volumes, modalities) + dimensions)
    labels = np.zeros((num_volumes, 1) + dimensions)

    for img_idx in range(num_volumes) :
        filename = dataset_path + path + pattern.format(str(img_idx + 1), inputs[0])
        image_data[img_idx, 0] = read_volume(filename)#[:, :, :, 0]
        
        filename = dataset_path + path + pattern.format(str(img_idx + 1), inputs[1])
        image_data[img_idx, 1] = read_volume(filename)#[:, :, :, 0]

        filename = dataset_path + path + pattern.format(img_idx + 1, inputs[1])
        labels[img_idx, 0] = read_volume(filename)[:, :, :, 0]

        image_data[img_idx, 1] = labels[img_idx, 0] != 0

    label_mapper = {0 : 0, 10 : 0, 150 : 1, 250 : 2}
    for key in label_mapper.keys() :
        labels[labels == key] = label_mapper[key]

    return image_data, labels

def read_IBSR18_dataset(dataset_path, dataset_info) :
    num_volumes = dataset_info['num_volumes']
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    inputs = dataset_info['inputs']

    image_data = np.zeros((num_volumes, modalities) + dimensions)
    labels = np.zeros((num_volumes, 1) + dimensions)

    for img_idx in range(num_volumes) :
        filename = dataset_path + path + pattern.format(img_idx + 1, inputs[0])
        image_data[img_idx, 0] = read_volume(filename)[:, :, :, 0]

        filename = dataset_path + path + pattern.format(img_idx + 1, inputs[1])
        labels[img_idx, 0] = read_volume(filename)[:, :, :, 0]

    return image_data, labels

def read_MICCAI2012_dataset(dataset_path, dataset_info) :
    num_volumes = dataset_info['num_volumes']
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    folder_names = dataset_info['folder_names']

    image_data = np.zeros((np.sum(num_volumes), modalities) + dimensions)
    labels = np.zeros((np.sum(num_volumes), 1) + dimensions)

    training_set = [1000, 1006, 1009, 1012, 1015, 1001, 1007,
        1010, 1013, 1017, 1002, 1008, 1011, 1014, 1036]

    testing_set = [1003, 1019, 1038, 1107, 1119, 1004, 1023, 1039, 1110, 1122, 1005,
        1024, 1101, 1113, 1125, 1018, 1025, 1104, 1116, 1128]

    for img_idx, image_name in enumerate(training_set) :
        filename = dataset_path + path + pattern[0].format(folder_names[0], image_name)
        image_data[img_idx, 0] = read_volume(filename)

        filename = dataset_path + path + pattern[1].format(folder_names[1], image_name)
        labels[img_idx, 0] = read_volume(filename)

        image_data[img_idx, 0] = np.multiply(image_data[img_idx, 0], labels[img_idx, 0] != 0)
        image_data[img_idx, 1] = labels[img_idx, 0] != 0

    for img_idx, image_name in enumerate(testing_set) :
        idx = img_idx + num_volumes[0]
        filename = dataset_path + path + pattern[0].format(folder_names[2], image_name)
        image_data[idx, 0] = read_volume(filename)

        filename = dataset_path + path + pattern[1].format(folder_names[3], image_name)
        labels[idx, 0] = read_volume(filename)

        image_data[idx, 0] = np.multiply(image_data[idx, 0], labels[idx, 0] != 0)
        image_data[idx, 1] = labels[idx, 0] != 0

    labels[labels > 4] = 0

    return image_data, labels

'''
    Read reconstructed results at evaluation step
'''
def read_result_volume_path(gen_conf, test_conf, case_name = None) :
    dataset = test_conf['dataset']
    if dataset in ['HCP', 'HCP-Wu-Minn-Contrast-Multimodal', 'HCP-Wu-Minn-Contrast-Multimodal-test',
                   'Nigeria19-Multimodal', 'Juntendo-Volunteer', 'MBB-rc', 'MBB-rc-test',  'HCP-rc',
                   'HCP-rc-test', 'ICH', 'Nigeria17-rc', 'Nigeria19-rc'] :
        return read_result_volume_Multimodal_path(gen_conf, test_conf, case_name)
    if dataset in ['HCP-LRTV', 'HCP-cubic', 'HCP-GT', 'HCP-input']:
        return read_result_HCP_single_contrast_path(gen_conf, test_conf, case_name)

def read_result_HCP_single_contrast_path(gen_conf, test_conf, case_name=None):
    if case_name is None:
        case_name = 'all'
    dataset = test_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    path = dataset_info['path'][0]
    pattern = dataset_info['general_pattern']
    in_postfix = dataset_info['in_postfix']

    # dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    modality_categories = dataset_info['modality_categories']
    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1]

    in_paths = []
    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       modality_categories[mod_idx],
                                                       in_postfix)
            in_paths.append(in_filename)

    return in_paths

def read_result_volume_Multimodal_path(gen_conf, test_conf, case_name = None) :
    if case_name is None:
        case_name = 'all'
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    modalities = dataset_info['modalities']
    modality_categories = dataset_info['modality_categories']
    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1]

    out_paths = []
    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            out_filename = generate_output_filename(
                gen_conf['results_path'],
                test_conf['dataset'],
                subject_lib[img_idx] + '_' + modality_categories[mod_idx] + '_c' + str(case_name),
                test_conf['approach'],
                test_conf['dimension'],
                str(test_conf['patch_shape']),
                str(test_conf['extraction_step']),
                dataset_info['format'])
            out_paths.append(out_filename)

    return out_paths

'''
    Clear reconstructed results at eval step
'''
def clear_result_volume(gen_conf, test_conf, case_name = None) :
    dataset = test_conf['dataset']
    if dataset in ['HCP', 'HCP-Wu-Minn-Contrast-Multimodal', 'HCP-Wu-Minn-Contrast-Multimodal-test',
                   'Nigeria19-Multimodal', 'Juntendo-Volunteer', 'MBB',  'HCP-rc', 'HCP-rc-test',
                   'MBB-rc', 'MBB-rc-test'] :
        return clear_result_volume_Multimodal(gen_conf, test_conf, case_name)

def clear_result_volume_Multimodal(gen_conf, test_conf, case_name = None) :
    if case_name is None:
        case_name = 'all'
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    modalities = dataset_info['modalities']
    modality_categories = dataset_info['modality_categories']
    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1]

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            out_filename = generate_output_filename(
                gen_conf['results_path'],
                test_conf['dataset'],
                subject_lib[img_idx] + '_' + modality_categories[mod_idx] + '_c' + str(case_name),
                test_conf['approach'],
                test_conf['dimension'],
                str(test_conf['patch_shape']),
                str(test_conf['extraction_step']),
                dataset_info['format'])
            os.remove(out_filename)

    return True

'''
    Read reconstructed results at evaluation step
'''
def read_result_volume(gen_conf, test_conf, case_name = None) :
    dataset = test_conf['dataset']
    if dataset == 'HCP-Wu-Minn-Contrast' :
        return read_result_volume_HCPWuMinnContrast(gen_conf, test_conf, case_name)
    elif dataset == 'HCP-Wu-Minn-Contrast-Augmentation' :
        return read_result_volume_HCPWuMinnContrastAugmentation(gen_conf, test_conf, case_name)
    elif dataset in ['HCP', 'HCP-Wu-Minn-Contrast-Multimodal', 'HCP-Wu-Minn-Contrast-Multimodal-test',
                     'Nigeria19-Multimodal', 'Juntendo-Volunteer', 'MBB', 'MBB-rc', 'MBB-rc-test',
                     'HCP-rc', 'HCP-rc-test', 'Nigeria17-rc', 'Nigeria19-rc'] :
        return read_result_volume_Multimodal(gen_conf, test_conf, case_name)
    ##  todo: template for the other datasets
    # elif dataset == 'MICCAI2012' :
    #     return save_volume_MICCAI2012(gen_conf, test_conf, volume, case_idx)
    # else:
    #     return save_volume_else(gen_conf, test_conf, volume, case_idx)

def read_result_volume_Multimodal(gen_conf, test_conf, case_name = None) :
    if case_name is None:
        case_name = 'all'
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    modality_categories = dataset_info['modality_categories']
    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1] # test sbjects

    out_data = np.zeros((num_volumes, modalities) + dimensions)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            out_filename = generate_output_filename(
                gen_conf['results_path'],
                test_conf['dataset'],
                subject_lib[img_idx] + '_' + modality_categories[mod_idx] + '_c' + str(case_name),
                test_conf['approach'],
                test_conf['dimension'],
                str(test_conf['patch_shape']),
                str(test_conf['extraction_step']),
                dataset_info['format'])
            out_data[img_idx, mod_idx] = read_volume(out_filename)

    # out_filename = generate_output_filename(
    #     gen_conf['results_path'],
    #     test_conf['dataset'],
    #     subject_lib[case_idx[0]]+'_'+modality_categories[case_idx[1]]+'_c'+str(case_idx[2]),
    #     test_conf['approach'],
    #     test_conf['dimension'],
    #     str(test_conf['patch_shape']),
    #     str(test_conf['extraction_step']),
    #     dataset_info['format'])

    return out_data

def read_result_volume_HCPWuMinnContrastAugmentation(gen_conf, test_conf, case_name = None) :
    if case_name is None:
        case_name = 'all'
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    modality_categories = dataset_info['modality_categories']
    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1]

    out_data = np.zeros((num_volumes, modalities) + dimensions)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            out_filename = generate_output_filename(
                gen_conf['results_path'],
                test_conf['dataset'],
                subject_lib[img_idx] + '_' + modality_categories[mod_idx] + '_c' + str(case_name),
                test_conf['approach'],
                test_conf['dimension'],
                str(test_conf['patch_shape']),
                str(test_conf['extraction_step']),
                dataset_info['format'])
            out_data[img_idx, mod_idx] = read_volume(out_filename)

    return out_data

def read_result_volume_HCPWuMinnContrast(gen_conf, test_conf, case_name = 1) :
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    modality_categories = dataset_info['modality_categories']
    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1]

    out_data = np.zeros((num_volumes, modalities) + dimensions)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            out_filename = generate_output_filename(
                gen_conf['results_path'],
                test_conf['dataset'],
                subject_lib[img_idx] + '_' + modality_categories[mod_idx] + '_c' + str(case_name),
                test_conf['approach'],
                test_conf['dimension'],
                str(test_conf['patch_shape']),
                str(test_conf['extraction_step']),
                dataset_info['format'])
            out_data[img_idx, mod_idx] = read_volume(out_filename)

    return out_data

def save_volume(gen_conf, test_conf, volume, case_idx) :
    dataset = test_conf['dataset']
    if   dataset == 'IBADAN-k8' :
        return save_volume_IBADAN(gen_conf, test_conf, volume, case_idx)
    elif dataset == 'HBN' :
        return save_volume_HBN(gen_conf, test_conf, volume, case_idx)
    elif dataset == 'HCP-Wu-Minn-Contrast' :
        return save_volume_HCPWuMinnContrast(gen_conf, test_conf, volume, case_idx)
    elif dataset == 'HCP-Wu-Minn-Contrast-Augmentation' :
        return save_volume_HCPWuMinnContrastAugmentation(gen_conf, test_conf, volume, case_idx)
    elif dataset in ['HCP', 'HCP-Wu-Minn-Contrast-Multimodal', 'HCP-Wu-Minn-Contrast-Multimodal-test',
                     'HCP-rc', 'HCP-rc-test', 'MBB-rc', 'MBB-rc-test'] :
        return save_volume_HCPWuMinnContrastMultimodal(gen_conf, test_conf, volume, case_idx)
    elif dataset == 'ICH' :
        return save_volume_ich(gen_conf, test_conf, volume, case_idx)
    elif dataset == 'Nigeria19-Multimodal' :
        return save_volume_Nigeria19Multimodal(gen_conf, test_conf, volume, case_idx)
    elif dataset == 'Juntendo-Volunteer' :
        return save_volume_JH_Volunteer(gen_conf, test_conf, volume, case_idx)
    elif dataset == 'MICCAI2012' :
        return save_volume_MICCAI2012(gen_conf, test_conf, volume, case_idx)
    elif dataset == 'MBB' :
        return save_volume_MBB(gen_conf, test_conf, volume, case_idx)
    elif dataset in ['Nigeria19-rc', 'Nigeria17-rc'] :
        return save_volume_N19_rc(gen_conf, test_conf, volume, case_idx)
    else:
        return save_volume_else(gen_conf, test_conf, volume, case_idx)

def save_volume_N19_rc(gen_conf, test_conf, volume, case_idx) :
    # todo: check conf and modalities
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    path = dataset_info['path'][0]
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    prefix = dataset_info['prefix']
    in_postfix = dataset_info['in_postfix'][0]
    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1]
    sparse_scale = dataset_info['sparse_scale']
    ext = dataset_info['format']

    if len(case_idx) > 3:
        case_idx[2] = str(str(case_idx[2])+'_'+str(case_idx[3]))

    data_filename = os.path.join(dataset_path,
                                 path,
                                 pattern).format(subject_lib[case_idx[0]],
                                                 prefix,
                                                 subject_lib[case_idx[0]][:3],
                                                 modality_categories[case_idx[1]],
                                                 in_postfix,
                                                 ext)
    image_data = read_volume_data(data_filename)

    ## change affine in the nii header
    if sparse_scale is not None:
        assert len(sparse_scale) == 3, "The length of sparse_scale is not equal to 3."
        # print image_data.affine
        nifty_affine = np.dot(image_data.affine, np.diag(tuple(1.0/np.array(sparse_scale))+(1.0, )))
        # print nifty_affine

    ## check and make folder
    out_filename = generate_output_filename(
        gen_conf['results_path'],
        test_conf['dataset'],
        subject_lib[case_idx[0]]+'_'+modality_categories[case_idx[1]]+'_c'+str(case_idx[2]),
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        dataset_info['format'])
    out_foldername = os.path.dirname(out_filename)
    if not os.path.isdir(out_foldername) :
        os.makedirs(out_foldername)
    print("Save file at {}".format(out_filename))
    __save_volume(volume, nifty_affine, out_filename, dataset_info['format'])

def save_volume_MBB(gen_conf, test_conf, volume, case_idx) :
    # todo: check conf and modalities
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    path = dataset_info['path'][0]
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    # in_postfix = dataset_info['in_postfix']

    if isinstance(case_idx, tuple):
        case_idx = list(case_idx)

    if case_idx[2] == 'ave':
        # case_idx[2] = 'all' # i_dataset
        # bugs here!!! 10/14
        in_postfixes = np.array(dataset_info['in_postfix'])
        in_postfix = in_postfixes[(0,) * in_postfixes.ndim]
    elif case_idx[2] == 'ave|rc' :
        in_postfix = np.array(dataset_info['in_postfix'])[0]
    else:
        print("case_idx:", case_idx)
        in_postfix = np.array(dataset_info['in_postfix'])[case_idx[2]]

    if len(case_idx) > 3:
        case_idx[2] = str(str(case_idx[2])+'_'+str(case_idx[3]))

    subject_lib = dataset_info['test_subjects']
    # num_volumes = dataset_info['num_volumes'][1]
    sparse_scale = dataset_info['sparse_scale']

    data_filename = os.path.join(dataset_path,
                                 path,
                                 pattern).format(subject_lib[case_idx[0]],
                                                 in_postfix)
    image_data = read_volume_data(data_filename)

    ## change affine in the nii header
    if sparse_scale is not None:
        assert len(sparse_scale) == 3, "The length of sparse_scale is not equal to 3."
        # print image_data.affine
        nifty_affine = np.dot(image_data.affine, np.diag(tuple(1.0/np.array(sparse_scale))+(1.0, )))
        # print nifty_affine

    ## check and make folder
    out_filename = generate_output_filename(
        gen_conf['results_path'],
        test_conf['dataset'],
        subject_lib[case_idx[0]]+'_'+modality_categories[case_idx[1]]+'_c'+str(case_idx[2]),
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        dataset_info['format'])
    out_foldername = os.path.dirname(out_filename)
    if not os.path.isdir(out_foldername) :
        os.makedirs(out_foldername)
    print("Save file at {}".format(out_filename))
    __save_volume(volume, nifty_affine, out_filename, dataset_info['format'])

def save_volume_JH_Volunteer(gen_conf, test_conf, volume, case_idx) :
    # todo: check conf and modalities
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    in_postfixes = dataset_info['in_postfixes']
    in_subfolder = dataset_info['subfolders'][0]
    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1]
    sparse_scale = dataset_info['sparse_scale']

    data_filename = os.path.join(dataset_path,
                                 path,
                                 pattern).format(in_subfolder,
                                                 subject_lib[case_idx[0]],
                                                 in_postfixes[case_idx[1]])
    image_data = read_volume_data(data_filename)

    ## change affine in the nii header
    if sparse_scale is not None:
        assert len(sparse_scale) == 3, "The length of sparse_scale is not equal to 3."
        # print image_data.affine
        nifty_affine = np.dot(image_data.affine, np.diag(tuple(1.0/np.array(sparse_scale))+(1.0, )))
        # print nifty_affine

    ## check and make folder
    out_filename = generate_output_filename(
        gen_conf['results_path'],
        test_conf['dataset'],
        subject_lib[case_idx[0]]+'_'+modality_categories[case_idx[1]]+'_c'+str(case_idx[2]),
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        dataset_info['format'])
    out_foldername = os.path.dirname(out_filename)
    if not os.path.isdir(out_foldername) :
        os.makedirs(out_foldername)
    print("Save file at {}".format(out_filename))
    __save_volume(volume, nifty_affine, out_filename, dataset_info['format'])

def save_volume_ich(gen_conf, test_conf, volume, case_idx) :
    '''
    10/14: update for random contrast case
    :param gen_conf:
    :param test_conf:
    :param volume:
    :param case_idx: (sbj_id, mod_id, i_dataset), (sbj_id, mod_id, 'ave')
    :return:
    '''

    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    path = dataset_info['path'][0] # 'process' path
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']

    if case_idx[2] == 'ave':
        in_postfixes = np.array([dataset_info['in_postfix']])
        in_postfix = in_postfixes[(0,) * in_postfixes.ndim]
    elif case_idx[2] == 'ave|rc' :
        in_postfix = np.array([dataset_info['in_postfix']])[0]
    else:
        print("case_idx:", case_idx)
        in_postfix = np.array([dataset_info['in_postfix']])[case_idx[2]]

    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1]
    sparse_scale = dataset_info['sparse_scale']

    if in_postfix.ndim == 1:
        in_postfix_sub = in_postfix[case_idx[0]]
    else:
        in_postfix_sub = in_postfix
    data_filename = os.path.join(dataset_path,
                                 path,
                                 pattern).format(subject_lib[case_idx[0]],
                                                 modality_categories[case_idx[1]],
                                                 in_postfix_sub)
    image_data = read_volume_data(data_filename)

    ## change affine in the nii header
    if sparse_scale is not None:
        assert len(sparse_scale) == 3, "The length of sparse_scale is not equal to 3."
        # print image_data.affine
        nifty_affine = np.dot(image_data.affine, np.diag(tuple(1.0/np.array(sparse_scale))+(1.0, )))
        # print nifty_affine

    ## check and make folder
    out_filename = generate_output_filename(
        gen_conf['results_path'],
        test_conf['dataset'],
        subject_lib[case_idx[0]]+'_'+modality_categories[case_idx[1]]+'_c'+str(case_idx[2]),
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        dataset_info['format'])
    out_foldername = os.path.dirname(out_filename)
    if not os.path.isdir(out_foldername) :
        os.makedirs(out_foldername)
    print("Save file at {}".format(out_filename))
    __save_volume(volume, nifty_affine, out_filename, dataset_info['format'])

def save_volume_HCPWuMinnContrastMultimodal(gen_conf, test_conf, volume, case_idx) :
    '''
    10/14: update for random contrast case
    :param gen_conf:
    :param test_conf:
    :param volume:
    :param case_idx: (sbj_id, mod_id, i_dataset, comments), (sbj_id, mod_id, 'ave', comments) if any comments
    :return:
    '''

    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    path = dataset_info['path'][0] # 'process' path
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']

    if isinstance(case_idx, tuple):
        case_idx = list(case_idx)

    if case_idx[2] == 'ave':
        # case_idx[2] = 'all' # i_dataset
        # bugs here!!! 10/14
        in_postfixes = np.array(dataset_info['in_postfix'])
        in_postfix = in_postfixes[(0,) * in_postfixes.ndim]
    elif case_idx[2] == 'ave|rc' :
        in_postfix = np.array(dataset_info['in_postfix'])[0]
    else:
        print("case_idx:", case_idx)
        in_postfix = np.array(dataset_info['in_postfix'])[case_idx[2]]

    if len(case_idx) > 3:
        case_idx[2] = str(str(case_idx[2])+'_'+str(case_idx[3]))

    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1]
    sparse_scale = dataset_info['sparse_scale']

    if in_postfix.ndim == 1:
        in_postfix_sub = in_postfix[case_idx[0]]
    else:
        in_postfix_sub = in_postfix
    data_filename = os.path.join(dataset_path,
                                 path,
                                 pattern).format(subject_lib[case_idx[0]],
                                                 modality_categories[case_idx[1]],
                                                 in_postfix_sub)
    image_data = read_volume_data(data_filename)

    ## change affine in the nii header
    if sparse_scale is not None:
        assert len(sparse_scale) == 3, "The length of sparse_scale is not equal to 3."
        # print image_data.affine
        nifty_affine = np.dot(image_data.affine, np.diag(tuple(1.0/np.array(sparse_scale))+(1.0, )))
        # print nifty_affine

    ## check and make folder
    out_filename = generate_output_filename(
        gen_conf['results_path'],
        test_conf['dataset'],
        subject_lib[case_idx[0]]+'_'+modality_categories[case_idx[1]]+'_c'+str(case_idx[2]),
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        dataset_info['format'])
    out_foldername = os.path.dirname(out_filename)
    if not os.path.isdir(out_foldername) :
        os.makedirs(out_foldername)
    print("Save file at {}".format(out_filename))
    __save_volume(volume, nifty_affine, out_filename, dataset_info['format'])

def save_volume_Nigeria19Multimodal(gen_conf, test_conf, volume, case_idx) :
    # todo: check conf and modalities
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    prefix = dataset_info['prefix']
    in_postfix = dataset_info['in_postfix']
    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1]
    sparse_scale = dataset_info['sparse_scale']
    ext = dataset_info['format']

    data_filename = os.path.join(dataset_path,
                                 path,
                                 pattern).format(subject_lib[case_idx[0]],
                                                 prefix,
                                                 subject_lib[case_idx[0]][:3],
                                                 modality_categories[case_idx[1]],
                                                 in_postfix,
                                                 ext)
    image_data = read_volume_data(data_filename)

    ## change affine in the nii header
    if sparse_scale is not None:
        assert len(sparse_scale) == 3, "The length of sparse_scale is not equal to 3."
        # print image_data.affine
        nifty_affine = np.dot(image_data.affine, np.diag(tuple(1.0/np.array(sparse_scale))+(1.0, )))
        # print nifty_affine

    ## check and make folder
    out_filename = generate_output_filename(
        gen_conf['results_path'],
        test_conf['dataset'],
        subject_lib[case_idx[0]]+'_'+modality_categories[case_idx[1]]+'_c'+str(case_idx[2]),
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        dataset_info['format'])
    out_foldername = os.path.dirname(out_filename)
    if not os.path.isdir(out_foldername) :
        os.makedirs(out_foldername)
    print("Save file at {}".format(out_filename))
    __save_volume(volume, nifty_affine, out_filename, dataset_info['format'])


def save_volume_HCPWuMinnContrastAugmentation(gen_conf, test_conf, volume, case_idx) :
    # todo: check conf and modalities
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    in_postfix = dataset_info['test_in_postfix'] # for test stageg
    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1]
    sparse_scale = dataset_info['sparse_scale']

    data_filename = os.path.join(dataset_path,
                                 path,
                                 pattern).format(subject_lib[case_idx[0]],
                                                 modality_categories[case_idx[1]],
                                                 in_postfix[case_idx[0]][case_idx[1]][0])
    image_data = read_volume_data(data_filename)

    ## change affine in the nii header
    if sparse_scale is not None:
        assert len(sparse_scale) == 3, "The length of sparse_scale is not equal to 3."
        # print image_data.affine
        nifty_affine = np.dot(image_data.affine, np.diag(tuple(1.0/np.array(sparse_scale))+(1.0, )))
        # print nifty_affine

    ## check and make folder
    out_filename = generate_output_filename(
        gen_conf['results_path'],
        test_conf['dataset'],
        subject_lib[case_idx[0]]+'_'+modality_categories[case_idx[1]]+'_c'+str(case_idx[2]),
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        dataset_info['format'])
    out_foldername = os.path.dirname(out_filename)
    if not os.path.isdir(out_foldername) :
        os.makedirs(out_foldername)
    print("Save file at {}".format(out_filename))
    __save_volume(volume, nifty_affine, out_filename, dataset_info['format'])

def save_volume_IBADAN(gen_conf, test_conf, volume, case_idx) :
    # todo: check conf and modalities
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    in_postfix = dataset_info['in_postfix']
    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1]
    sparse_scale = dataset_info['sparse_scale']

    data_filename = os.path.join(dataset_path,
                                 path,
                                 pattern).format(subject_lib[case_idx[0]],
                                                 modality_categories[case_idx[1]],
                                                 in_postfix)
    image_data = read_volume_data(data_filename)

    ## change affine in the nii header
    if sparse_scale is not None:
        assert len(sparse_scale) == 3, "The length of sparse_scale is not equal to 3."
        # print image_data.affine
        nifty_affine = np.dot(image_data.affine, np.diag(tuple(1.0/np.array(sparse_scale))+(1.0, )))
        # print nifty_affine

    ## check and make folder
    out_filename = generate_output_filename(
        gen_conf['results_path'],
        test_conf['dataset'],
        subject_lib[case_idx[0]]+'_'+modality_categories[case_idx[1]],
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        dataset_info['format'])
    out_foldername = os.path.dirname(out_filename)
    if not os.path.isdir(out_foldername) :
        os.makedirs(out_foldername)
    print("Save file at {}".format(out_filename))
    __save_volume(volume, nifty_affine, out_filename, dataset_info['format'])

def save_volume_HBN(gen_conf, test_conf, volume, case_idx) :
    # todo: check conf and modalities
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    in_postfix = dataset_info['in_postfix']
    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1]
    sparse_scale = dataset_info['sparse_scale']

    data_filename = os.path.join(dataset_path,
                                 path,
                                 pattern).format(subject_lib[case_idx[0]], subject_lib[case_idx[0]],
                                                 modality_categories[case_idx[1]],
                                                 in_postfix)
    image_data = read_volume_data(data_filename)

    ## change affine in the nii header
    if sparse_scale is not None:
        assert len(sparse_scale) == 3, "The length of sparse_scale is not equal to 3."
        # print image_data.affine
        nifty_affine = np.dot(image_data.affine, np.diag(tuple(1.0/np.array(sparse_scale))+(1.0, )))
        # print nifty_affine

    ## check and make folder
    out_filename = generate_output_filename(
        gen_conf['results_path'],
        test_conf['dataset'],
        subject_lib[case_idx[0]]+'_'+modality_categories[case_idx[1]],
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        dataset_info['format'])
    out_foldername = os.path.dirname(out_filename)
    if not os.path.isdir(out_foldername) :
        os.makedirs(out_foldername)
    print("Save file at {}".format(out_filename))
    __save_volume(volume, nifty_affine, out_filename, dataset_info['format'])

def save_volume_HCPWuMinnContrast(gen_conf, test_conf, volume, case_idx) :
    # todo: check conf and modalities
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    in_postfix = dataset_info['in_postfix']
    subject_lib = dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][1]
    sparse_scale = dataset_info['sparse_scale']

    data_filename = os.path.join(dataset_path,
                                 path,
                                 pattern).format(subject_lib[case_idx[0]],
                                                 modality_categories[case_idx[1]],
                                                 in_postfix)
    image_data = read_volume_data(data_filename)

    ## change affine in the nii header
    if sparse_scale is not None:
        assert len(sparse_scale) == 3, "The length of sparse_scale is not equal to 3."
        # print image_data.affine
        nifty_affine = np.dot(image_data.affine, np.diag(tuple(1.0/np.array(sparse_scale))+(1.0, )))
        # print nifty_affine

    ## check and make folder
    out_filename = generate_output_filename(
        gen_conf['results_path'],
        test_conf['dataset'],
        subject_lib[case_idx[0]]+'_'+modality_categories[case_idx[1]]+'_'+str(case_idx[2]),
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        dataset_info['format'])
    out_foldername = os.path.dirname(out_filename)
    if not os.path.isdir(out_foldername) :
        os.makedirs(out_foldername)
    print("Save file at {}".format(out_filename))
    __save_volume(volume, nifty_affine, out_filename, dataset_info['format'])

def save_volume_MICCAI2012(gen_conf, train_conf, volume, case_idx) :
    dataset = train_conf['dataset']
    approach = train_conf['approach']
    extraction_step = train_conf['extraction_step_test']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    results_path = gen_conf['results_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    folder_names = dataset_info['folder_names']

    data_filename = dataset_path + path + pattern[1].format(folder_names[3], case_idx)
    image_data = read_volume_data(data_filename)

    volume = np.multiply(volume, image_data.get_data() != 0)

    out_filename = results_path + path + pattern[2].format(folder_names[3], str(case_idx), approach + ' - ' + str(extraction_step))

    __save_volume(volume, image_data.affine, out_filename, dataset_info['format'])

def save_volume_else(gen_conf, train_conf, volume, case_idx) :
    dataset = train_conf['dataset']
    approach = train_conf['approach']
    extraction_step = train_conf['extraction_step_test']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    results_path = gen_conf['results_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    inputs = dataset_info['inputs']

    if dataset == 'iSeg2017' or dataset == 'IBSR18':
        volume_tmp = np.zeros(volume.shape + (1, ))
        volume_tmp[:, :, :, 0] = volume
        volume = volume_tmp

    data_filename = dataset_path + path + pattern.format(case_idx, inputs[-1])
    image_data = read_volume_data(data_filename)

    volume = np.multiply(volume, image_data.get_data() != 0)

    if dataset == 'iSeg2017' :
        volume[image_data.get_data() != 0] = volume[image_data.get_data() != 0] + 1

        label_mapper = {0 : 0, 1 : 10, 2 : 150, 3 : 250}
        for key in label_mapper.keys() :
            volume[volume == key] = label_mapper[key]

    out_filename = results_path + path + pattern.format(case_idx, approach + ' - ' + str(extraction_step))

    ## mkdir
    if not os.path.isdir(os.path.dirname(out_filename)):
        os.makedirs(os.path.dirname(out_filename))

    __save_volume(volume, image_data.affine, out_filename, dataset_info['format'])

def __save_volume(volume, nifty_affine, filename, format) :
    img = None
    if format == 'nii' or format == 'nii.gz' :
        img = nib.Nifti1Image(volume.astype('float32'), nifty_affine) # uint8
    if format == 'analyze' :
        img = nib.analyze.AnalyzeImage(volume.astype('float32'), nifty_affine) # uint8
    nib.save(img, filename)

def read_volume(filename) :
    return read_volume_data(filename).get_data()

def read_volume_data(filename) :
    return nib.load(filename)

def generate_output_filename(
    path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension) :
#     file_pattern = '{}/{}/{:02}-{}-{}-{}-{}.{}'
    file_pattern = '{}/{}/{}-{}-{}-{}-{}.{}'
    print(file_pattern.format(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension))
    return file_pattern.format(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension)

'''
    Read patch data
'''
def read_patch_data(filepath) :
    files = np.load(filepath).files
    return files['x_patches'], files['y_patches']

'''
    Save patch data with the input-image file name but with the '.npz' postfix 
'''
def save_patch_data(x_patches, y_patches, filepath) :
    np.savez(filepath, x_patches=x_patches, y_patches=y_patches)
    return True

def generate_patch_filename( modality, sample_num, patch_shape, extraction_step, extension = 'npz') :
    file_pattern = '{}-{}-{}-{}.{}'
    print(file_pattern.format( modality, sample_num, patch_shape, extraction_step, extension))
    return file_pattern.format( modality, sample_num, patch_shape, extraction_step, extension)