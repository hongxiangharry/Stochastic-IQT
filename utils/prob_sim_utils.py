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
""" Utilities for probabilistic low-field MRI simulation. """

import os
import csv
from shutil import copy2
import itertools
import numpy as np
from numpy.random import multivariate_normal, seed
from scipy.stats import multivariate_normal as multi_normal
from utils.ioutils import read_data_path
from utils.mri_simulator import MRI_036T_sim_contrast_custom


def generate_sim(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag = 'train'):
    if 'contrast_type' not in sim_norm_conf['sim'].keys():
        print('Use contrast type as mc ..')
        sim_norm_conf['sim']['contrast_type'] = 'mc'

    if sim_norm_conf['sim']['contrast_type'] == 'mc' or sim_norm_conf['sim']['contrast_type'] == 'multi-contrast':
        print('Run simulation on MC ...')
        gen_conf, traintest_conf, sim_norm_conf = generate_sim_multi_contrast\
            (gen_conf, traintest_conf, sim_norm_conf, trainTestFlag)
    elif sim_norm_conf['sim']['contrast_type'] == 'rc' or sim_norm_conf['sim']['contrast_type'] == 'random-contrast':
        print('Run simulation on RC ...')
        gen_conf, traintest_conf, sim_norm_conf = generate_sim_with_random_contrast\
            (gen_conf, traintest_conf, sim_norm_conf, trainTestFlag)
    else:
        raise Exception("contrast type has not been defined ...")

    return gen_conf, traintest_conf, sim_norm_conf


def generate_sim_multi_contrast(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag = 'train', normType='separate'):
    # simulation
    gen_conf, traintest_conf, sim_norm_conf = \
        update_all_confs_after_sim(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag=trainTestFlag) # generate sim params
    sim_conf = sim_norm_conf['sim']
    if sim_conf['is_sim'] is False:
        generate_sim_data(gen_conf, traintest_conf, sim_conf, trainTestFlag=trainTestFlag)

    return gen_conf, traintest_conf, sim_norm_conf

def generate_sim_with_random_contrast(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag = 'train'):
    # simulation
    gen_conf, traintest_conf, sim_norm_conf = \
        update_all_confs_after_sim_with_random_contrast(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag=trainTestFlag) # generate sim params
    sim_conf = sim_norm_conf['sim']
    if sim_conf['is_sim'] is False:
        generate_all_sim_data_with_random_contrast(gen_conf, traintest_conf, sim_conf, trainTestFlag=trainTestFlag)

    return gen_conf, traintest_conf, sim_norm_conf

def __generate_sim_params(sim_conf) :
    distribution = sim_conf['distribution'] # name of distribution
    if 'seed' in sim_conf:
        seed(sim_conf['seed'])

    if distribution == 'point':
        assert 'sim_params' in sim_conf, "'sim_params' doesn't exist in sim_conf"
        assert 'sim_pdf' in sim_conf, "'sim_pdf' doesn't exist in sim_conf"
    else:
        params = sim_conf['params']
        if distribution == 'gaussian':
            n_samples = sim_conf['n_samples']  # num of sample params for training and eval
            sim_conf['sim_params'] = multivariate_normal(params['mean'], params['cov'], size=n_samples) # return #n_samples x #sim params
            sim_conf['sim_pdf'] = multi_normal.pdf(sim_conf['sim_params'], mean = params['mean'], cov = params['cov'])

    return sim_conf

def update_all_confs_after_sim(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag='train'):
    '''
    generate all configs after sampling or defining simulated parameters.
    NOTE: 1. modalities == n_sample? this is a historic issue. I left an entry for multimodal IQT, however it was used in a multi-contrast case afterward
    2. for multi-contrast case
    '''

    # update sim_conf
    sim_conf = sim_norm_conf['sim']
    sim_conf = __generate_sim_params(sim_conf)
    sim_params = sim_conf['sim_params']
    print("Simulation params: ", sim_params)

    # update gen_conf -> dataset_info -> in_postfix, out_postfix, 'modality_categories'

    dataset = traintest_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_info['in_postfix'] = []
    in_postfix_pattern = dataset_info['in_postfix_pattern']
    dataset_info['out_postfix'] = dataset_info['out_postfix_pattern'].format(dataset_info['downsample_scale'])

    norm_conf = sim_norm_conf['norm']
    if 'normType' not in norm_conf.keys():
        normType = None
    else:
        normType = norm_conf['normType']

    if normType is None or normType == 'separate':
        traintest_conf['num_dataset'] = sim_conf['n_samples_train']  # update number of sim-norm
    elif normType == 'merge':
        traintest_conf['num_dataset'] = 1  # update number of sim-norm

    if trainTestFlag == 'train':
        for idx in range(sim_conf['n_samples_train']):
            suffix = '_WM{:.2f}_GM{:.2f}'.format(sim_params[idx][0], sim_params[idx][1])
            dataset_info['in_postfix'].append(in_postfix_pattern.format(dataset_info['downsample_scale'], suffix))

        save_sim_info(gen_conf, traintest_conf, sim_conf)

    elif trainTestFlag == 'eval':
        traintest_conf['num_test_dataset'] = len(sim_params) # update number of test sim params
        for idx in range(traintest_conf['num_dataset']): # train
            in_postfix_temp = []
            for idx2 in range(traintest_conf['num_test_dataset']): # test
                suffix = '_WM{:.2f}_GM{:.2f}_{}'.format(sim_params[idx2][0], sim_params[idx2][1], idx)
                in_postfix_temp.append(in_postfix_pattern.format(dataset_info['downsample_scale'], suffix))
            dataset_info['in_postfix'].append(in_postfix_temp)
    elif trainTestFlag == 'test':
        traintest_conf['num_test_dataset'] = len(sim_params)  # update number of test sim params
        for idx in range(traintest_conf['num_test_dataset']): # test
            suffix = '_{}'.format(idx)
            dataset_info['in_postfix'].append(in_postfix_pattern.format(dataset_info['downsample_scale'], suffix))

    # save into the old confs
    gen_conf['dataset_info'][dataset] = dataset_info
    sim_norm_conf['sim'] = sim_conf

    return gen_conf, traintest_conf, sim_norm_conf

def broadcast_list(original_list, broadcast_dim):
    '''
    assert len(original_list) <= broadcast_dim
    '''
    original_dim = len(original_list)
    assert original_dim <= broadcast_dim, "Please ensure the sim param dimension <= broadcast dimension."
    quo = broadcast_dim//original_dim # quotient
    rem = broadcast_dim % original_dim # remainder
    if rem != 0:
        repeats_arr = np.repeat(quo, original_dim, axis=0)
        repeats_arr[-1] = repeats_arr[-1] + rem
    else:
        repeats_arr = np.repeat(quo, original_dim, axis=0)
    b_list = np.repeat(original_list, repeats_arr, axis=0)
    return b_list

def update_all_confs_after_sim_with_random_contrast(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag='train'):
    ''' generate all configs after sampling or defining simulated parameters.
    NOTE:
        1. modalities == n_sample? this is a historic issue. I left an entry for multimodal IQT, however it was used in a multi-contrast case afterward
        2. If running multi-contrast IQT at test time, i.e. not all test have same contrast at test or we don't have to simulate all contrast at evaluation time, we need to expand sim_params and in_postfix as long as test_sbjs (9/24)
    Params:
        num_volumes: train/test subjects number
        num_dataset: train model number
        num_test_dataset: num SNR/subject, here 1
    '''

    # update sim_norm_conf
    ## only support merge normalization and one model
    ## sim_conf['n_samples'] is the number of sim params for sampling.
    ## Assume num_dataset is specified.
    ## For training, traintest_conf['num_dataset'] == #train_models == sim_conf['n_samples']
    ## For test, traintest_conf['num_dataset'] should still represent #train_models
    ## For test, sim_params = test_volumes if not specified
    sim_conf = sim_norm_conf['sim']
    norm_conf = sim_norm_conf['norm']
    if 'normType' not in norm_conf.keys(): # only support 'merge' for random contrast IQT
        normType = 'merge'
    else:
        normType = norm_conf['normType']
    assert normType == 'merge', 'Only support merge normalization for random contrast IQT so check sim_norm_conf.'

    traintest_conf['num_dataset'] = 1  # naming error, this should indicate number of trained models

    dataset = traintest_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_info['in_postfix'] = []
    in_postfix_pattern = dataset_info['in_postfix_pattern']
    dataset_info['out_postfix'] = dataset_info['out_postfix_pattern'].format(dataset_info['downsample_scale'])

    # generate and expand sim_param
    if trainTestFlag == 'train':
        num_volumes = dataset_info['num_volumes'][0]
        if 'num_train_sbj' in traintest_conf.keys():
            num_volumes = traintest_conf['num_train_sbj']
        sim_conf['n_samples'] = traintest_conf['num_train_sbj']
    elif trainTestFlag == 'eval' or trainTestFlag == 'test':
        num_volumes = dataset_info['num_volumes'][1]
        sim_conf['n_samples'] = num_volumes

    sim_conf = __generate_sim_params(sim_conf)
    sim_conf['sim_params'] = broadcast_list(sim_conf['sim_params'], num_volumes) # bug?
    sim_params = sim_conf['sim_params']

    # expand in_postfix
    if trainTestFlag == 'train':
        # update 12/21: add contrast_indices to identify

        for idx in range(sim_conf['n_samples']):
            suffix = '_WM{:.2f}_GM{:.2f}'.format(sim_params[idx][0], sim_params[idx][1])
            dataset_info['in_postfix'].append(in_postfix_pattern.format(dataset_info['downsample_scale'], suffix))

        # broadcasting
        dataset_info['in_postfix'] = broadcast_list(dataset_info['in_postfix'], num_volumes)
        sim_conf['contrast_indices'] = broadcast_list(np.arange(sim_conf['n_samples']), num_volumes)

        save_sim_info(gen_conf, traintest_conf, sim_conf)

    elif trainTestFlag == 'eval':
        traintest_conf['num_test_dataset'] = 1 # update number of test sim params
        ## update: num_test_dataset != num_test_data

        for idx in range(traintest_conf['num_dataset']): # train
            in_postfix_temp = []
            for idx2 in range(num_volumes): # test
                suffix = '_WM{:.2f}_GM{:.2f}_{}'.format(sim_params[idx2][0], sim_params[idx2][1], idx)
                in_postfix_temp.append(in_postfix_pattern.format(dataset_info['downsample_scale'], suffix))
            dataset_info['in_postfix'].append(in_postfix_temp)
        print("Print in_postfix", dataset_info['in_postfix'])
    elif trainTestFlag == 'test':
        traintest_conf['num_test_dataset'] = 1  # update number of test sim params
        for idx in range(num_volumes): # test
            suffix = '_{}'.format(idx)
            dataset_info['in_postfix'].append(in_postfix_pattern.format(dataset_info['downsample_scale'], suffix))

    # save into the old confs
    gen_conf['dataset_info'][dataset] = dataset_info
    sim_norm_conf['sim'] = sim_conf

    return gen_conf, traintest_conf, sim_norm_conf

def generate_sim_data(gen_conf, traintest_conf, sim_conf, trainTestFlag = 'train'):
    dataset = traintest_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    i_shape_org = None
    if 'is_upsample' in dataset_info.keys():
        is_upsample = dataset_info['is_upsample']
    else:
        is_upsample = False

    # sim
    sim_params = sim_conf['sim_params']
    stf = sim_conf['slice_thickness_factor']
    gap = sim_conf['gap']

    _, hf_origin_paths = read_data_path(gen_conf, traintest_conf, trainTestFlag, originProcessFlag='origin')

    if trainTestFlag == 'train':
        num_sim_train = traintest_conf['num_dataset'] # num of train models = # num of contrasts = num of SNRs
        iters = range(num_sim_train)
    elif trainTestFlag == 'eval':
        num_sim_train = traintest_conf['num_dataset']
        num_sim_test = traintest_conf['num_test_dataset'] # should = num_volumes
        iters = itertools.product(range(num_sim_train), range(num_sim_test))

    # simulate data so as to have the same contrast in LF and HF
    for i_dataset in iters:
        lf_process_paths, hf_process_paths = read_data_path(gen_conf, traintest_conf, trainTestFlag, originProcessFlag='process', i_dataset=i_dataset) ## select SNR for all paths

        if isinstance(i_dataset, tuple):
            i_dataset = i_dataset[1] # eval
        sim_param = np.array(sim_params)[i_dataset]

        for hf_origin_path, lf_process_path, hf_process_path in zip(hf_origin_paths, lf_process_paths, hf_process_paths):
            print("Simulating: ", lf_process_path)
            MRI_036T_sim_contrast_custom(
                os.path.dirname(hf_origin_path),
                os.path.basename(hf_origin_path),
                os.path.dirname(lf_process_path),
                os.path.basename(lf_process_path),
                sim_param[0],
                sim_param[1],
                stf,
                gap,
                savegroundtruth_path=os.path.basename(hf_process_path),
                i_shape_org=i_shape_org,
                is_upsample=is_upsample
            )

    return True

def generate_all_sim_data_with_random_contrast(gen_conf, traintest_conf, sim_conf, trainTestFlag = 'train'):
    '''
    general all sim data with random contrast
    '''

    dataset = traintest_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    i_shape_org = None
    if 'is_upsample' in dataset_info.keys():
        is_upsample = dataset_info['is_upsample']
    else:
        is_upsample = False

    if 'is_groundtruth_masked' in dataset_info.keys():
        is_save_masked_groundtruth = not dataset_info['is_groundtruth_masked']
    else:
        is_save_masked_groundtruth = False # by default, data are all skull striped

    if 'is_sim_WM_fixed' in dataset_info.keys():
        is_sim_WM_fixed = dataset_info['is_sim_WM_fixed']
    else:
        is_sim_WM_fixed = True

    # sim
    sim_params = sim_conf['sim_params']
    stf = sim_conf['slice_thickness_factor']
    gap = sim_conf['gap']

    if trainTestFlag == 'train':
        num_volumes = dataset_info['num_volumes'][0]
        if 'num_train_sbj' in traintest_conf.keys():
            num_volumes = traintest_conf['num_train_sbj'] # 60 by default
        iters = range(num_volumes)

        # simulate data so as to have the same contrast in LF and HF
        for i_dataset in iters:
            if isinstance(i_dataset, tuple):
                idx = i_dataset[1]  # eval
            else:
                idx = i_dataset

            sim_param = np.array(sim_params)[idx]

            _, hf_origin_paths = read_data_path(gen_conf, traintest_conf, trainTestFlag, originProcessFlag='origin',
                                                i_dataset=i_dataset, indices=[idx, 0])
            lf_process_paths, hf_process_paths = read_data_path(gen_conf, traintest_conf, trainTestFlag,
                                                                originProcessFlag='process', i_dataset=i_dataset,
                                                                indices=[idx, 0])  ## select SNR for all paths

            for hf_origin_path, lf_process_path, hf_process_path in zip(hf_origin_paths, lf_process_paths,
                                                                        hf_process_paths):
                print("Simulating: ", lf_process_path)
                MRI_036T_sim_contrast_custom(
                    os.path.dirname(hf_origin_path),
                    os.path.basename(hf_origin_path),
                    os.path.dirname(lf_process_path),
                    os.path.basename(lf_process_path),
                    sim_param[0],
                    sim_param[1],
                    stf,
                    gap,
                    savegroundtruth_path=os.path.basename(hf_process_path),
                    i_shape_org=i_shape_org,
                    is_upsample=is_upsample,
                    is_save_masked_groundtruth=is_save_masked_groundtruth,
                    is_sim_WM_fixed=is_sim_WM_fixed
                )

    elif trainTestFlag == 'eval':
        num_volumes = dataset_info['num_volumes'][1]
        num_sim_train = traintest_conf['num_dataset']
        iters = itertools.product(range(num_sim_train), range(num_volumes))

        # simulate data so as to have the same contrast in LF and HF
        for id_train, idx in iters:

            sim_param = np.array(sim_params)[idx]

            _, hf_origin_paths = read_data_path(gen_conf, traintest_conf, trainTestFlag, originProcessFlag='origin', i_dataset=(id_train, idx), indices=[idx, 0])
            lf_process_paths, hf_process_paths = read_data_path(gen_conf, traintest_conf, trainTestFlag, originProcessFlag='process', i_dataset=(id_train, idx), indices=[idx, 0]) ## select SNR for all paths

            for hf_origin_path, lf_process_path, hf_process_path in zip(hf_origin_paths, lf_process_paths, hf_process_paths):
                if id_train == 0:
                    print("Simulating: ", lf_process_path)
                    MRI_036T_sim_contrast_custom(
                        os.path.dirname(hf_origin_path),
                        os.path.basename(hf_origin_path),
                        os.path.dirname(lf_process_path),
                        os.path.basename(lf_process_path),
                        sim_param[0],
                        sim_param[1],
                        stf,
                        gap,
                        savegroundtruth_path=os.path.basename(hf_process_path),
                        i_shape_org=i_shape_org,
                        is_upsample=is_upsample,
                        is_save_masked_groundtruth=is_save_masked_groundtruth,
                        is_sim_WM_fixed=is_sim_WM_fixed
                    )
                else:
                    lf_process_paths0, _ = read_data_path(gen_conf, traintest_conf, trainTestFlag, originProcessFlag='process', i_dataset=(0, idx), indices=[idx, 0])  ## select SNR for all paths
                    for lf_process_path0 in lf_process_paths0:
                        copy2(lf_process_path0, lf_process_path)


    return True

def save_sim_info(gen_conf, train_conf, sim_conf):
    dataset_name = train_conf['dataset']

    # check and create parent folder
    csv_filename_sim = generate_output_filename(
        gen_conf['log_path'],
        train_conf['dataset'],
        '_sim_conf',
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'csv')
    ## check and make folders
    csv_foldername = os.path.dirname(csv_filename_sim)
    if not os.path.isdir(csv_foldername) :
        os.makedirs(csv_foldername)

    # save gen_conf
    with open(csv_filename_sim, 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, sim_conf.keys())
        w.writeheader()
        w.writerow(sim_conf)

    return True


def generate_output_filename(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension):
    file_pattern = '{}/{}/{}-{}-{}-{}-{}.{}'
    return file_pattern.format(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension)