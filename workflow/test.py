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
""" Workflow for test stage. """

import os
import numpy as np
import itertools
import time
from utils.ioutils import read_dataset, save_volume, read_model, read_result_volume, clear_result_volume, read_data_path, read_volume, read_result_volume_path
from utils.ioutils import read_dataloader_meanstd
from utils.reconstruction import reconstruct_volume_imaging
from utils.patching_utils import overlap_patching

def test_contrast(gen_conf, test_conf, sim_norm_conf, train_conf=None, trainTestFlag='eval', isClearSingleOutputs=False):
    if 'contrast_type' not in sim_norm_conf['sim'].keys():
        print('Use contrast type as mc ..')
        sim_norm_conf['sim']['contrast_type'] = 'mc'

    if sim_norm_conf['sim']['contrast_type'] == 'mc' or sim_norm_conf['sim']['contrast_type'] == 'multi-contrast':
        print('Run test on MC ...')
        test_multi_contrast(gen_conf, test_conf, train_conf=train_conf, trainTestFlag=trainTestFlag, isClearSingleOutputs=isClearSingleOutputs)
    elif sim_norm_conf['sim']['contrast_type'] == 'rc' or sim_norm_conf['sim']['contrast_type'] == 'random-contrast':
        print('Run test on RC ...')
        test_random_contrast(gen_conf, test_conf, train_conf=train_conf, trainTestFlag=trainTestFlag, isClearSingleOutputs=isClearSingleOutputs)
    else:
        raise Exception("contrast type has not been defined ...")

    return True

def test_contrast_uncertainty(gen_conf, test_conf, sim_norm_conf, train_conf=None, trainTestFlag='eval', isClearSingleOutputs=False, is_agg=True):
    if 'contrast_type' not in sim_norm_conf['sim'].keys():
        print('Use contrast type as mc ..')
        sim_norm_conf['sim']['contrast_type'] = 'mc'

    if sim_norm_conf['sim']['contrast_type'] == 'rc' or sim_norm_conf['sim']['contrast_type'] == 'random-contrast':
        print('Run test on RC ...')
        test_random_contrast_uncertainty(gen_conf, test_conf, train_conf=train_conf, trainTestFlag=trainTestFlag, isClearSingleOutputs=isClearSingleOutputs, is_agg=is_agg)
    else:
        raise Exception("contrast type has not been defined ...")

    return True

'''
    for batchensembles
'''
def test_random_contrast_uncertainty(gen_conf, test_conf, train_conf=None, trainTestFlag='eval', isClearSingleOutputs=False, is_lowcost = True, is_agg = True):
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]

    print("in postfix:", dataset_info['in_postfix'])

    if trainTestFlag == 'test':
        num_sim_train = test_conf['num_dataset']
        iters = range(num_sim_train)
    elif trainTestFlag == 'eval':
        num_sim_train = test_conf['num_dataset'] # num train models
        iters = range(num_sim_train)

    # temporarily comment out todo: comment back!
    for i_dataset in iters:
        testing_uncertain_model_random_contrast(gen_conf, test_conf, train_conf, i_dataset=i_dataset, is_lowcost=is_lowcost)

    if is_lowcost is True and is_agg is True:
        aggregate_all_models_lowcost_uncertainty(gen_conf, test_conf, train_conf, trainTestFlag=trainTestFlag, isClearSingleOutputs=isClearSingleOutputs, contrast_type='rc')

    return True

def test_random_contrast(gen_conf, test_conf, train_conf=None, trainTestFlag='eval', isClearSingleOutputs=False):
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]

    print("in postfix:", dataset_info['in_postfix'])

    if trainTestFlag == 'test':
        num_sim_train = test_conf['num_dataset']
        iters = range(num_sim_train)
    elif trainTestFlag == 'eval':
        num_sim_train = test_conf['num_dataset'] # num train models
        iters = range(num_sim_train)

    is_lowcost = True

    for i_dataset in iters:
        testing_single_model_random_contrast(gen_conf, test_conf, train_conf, i_dataset=i_dataset, is_lowcost=is_lowcost)

    if is_lowcost is True:
        aggregate_all_models_lowcost(gen_conf, test_conf, train_conf, trainTestFlag=trainTestFlag, isClearSingleOutputs=isClearSingleOutputs, contrast_type='rc')
    else:
        aggregate_all_models(gen_conf, test_conf, train_conf, trainTestFlag=trainTestFlag, isClearSingleOutputs=isClearSingleOutputs, contrast_type='rc')

    return True

def test_multi_contrast(gen_conf, test_conf, train_conf = None, trainTestFlag='eval', isClearSingleOutputs=False):
    if trainTestFlag == 'test':
        num_sim_train = test_conf['num_dataset']
        iters = range(num_sim_train)
    elif trainTestFlag == 'eval':
        num_sim_train = test_conf['num_dataset']
        num_sim_test = test_conf['num_test_dataset']
        iters = itertools.product(range(num_sim_train), range(num_sim_test))

    is_lowcost = True

    for i_dataset in iters:
        testing_single_model_multi_contrast(gen_conf, test_conf, train_conf, i_dataset=i_dataset, is_lowcost=is_lowcost)

    if is_lowcost is True:
        aggregate_all_models_lowcost(gen_conf, test_conf, train_conf, trainTestFlag=trainTestFlag, isClearSingleOutputs=isClearSingleOutputs, contrast_type='mc')
    else:
        aggregate_all_models(gen_conf, test_conf, train_conf, trainTestFlag=trainTestFlag, isClearSingleOutputs=isClearSingleOutputs, contrast_type='mc')

    return True

def aggregate_all_models_lowcost_uncertainty(gen_conf, test_conf, train_conf = None, trainTestFlag='eval', isClearSingleOutputs=False, contrast_type='mc'):
    '''
    update:
    1. (23/05/15) Remove mri_dims and load mri dim from volumes input
    '''
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_test_sbj = dataset_info['num_volumes'][1] # number of test subjects
    num_modalities = dataset_info['modalities']

    if trainTestFlag == 'test':
        num_sim_train = test_conf['num_dataset']
        iters = range(num_sim_train)
        num_all_data = num_sim_train
    elif trainTestFlag == 'eval':
        num_sim_train = test_conf['num_dataset']

        if contrast_type == 'mc':
            num_sim_test = test_conf['num_test_dataset']
            iters = itertools.product(range(num_sim_train), range(num_sim_test))
        elif contrast_type == 'rc':
             iters = range(num_sim_train)
             num_sim_test = 1 # was test_conf['num_test_dataset']
        else:
            print('Error!')
        num_all_data = num_sim_train*num_sim_test
    else :
        raise ValueError("trainTestFlag should be declared as 'test'/'eval'")

    # for random-contrast: num_sim_test should be 1 !!
    for idx in range(num_test_sbj):
        for idx2 in range(num_modalities):
            for idx3, i_dataset in enumerate(iters):
                im_recon_path = read_result_volume_path(gen_conf, test_conf, str(i_dataset)+'{}') # load reconstructed image, shape : (subject, modality, mri_shape)
                im_recon_mean = read_volume(im_recon_path[idx*num_modalities+idx2].format(''))
                im_recon_std = read_volume(im_recon_path[idx * num_modalities + idx2].format('_std'))
                # initialise agg_im_mean and agg_im_std
                if idx3 == 0:
                    agg_im_mean = np.zeros_like(im_recon_mean)
                    agg_im_std = np.zeros_like(im_recon_std)
                if test_conf['ensemble_type'] == 'ave' or test_conf['ensemble_type'] == 'ave|rc':
                    agg_im_mean += im_recon_mean / num_all_data
                    agg_im_std += im_recon_std / num_all_data

            # save agg im
            save_volume(gen_conf, test_conf, agg_im_mean, (idx, idx2, test_conf['ensemble_type'])) # save agg im
            save_volume(gen_conf, test_conf, agg_im_std, (idx, idx2, test_conf['ensemble_type'], 'std'))  # save agg im

    for i_dataset in iters:
        ## clean single model outputs
        if isClearSingleOutputs is True:
            clear_result_volume(gen_conf, test_conf, i_dataset)

    return True

def aggregate_all_models_lowcost(gen_conf, test_conf, train_conf = None, trainTestFlag='eval', isClearSingleOutputs=False, contrast_type='mc'):
    '''
    update:
    1. (23/05/15) Remove mri_dims and load mri dim from volumes input
    '''
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_test_sbj = dataset_info['num_volumes'][1] # number of test subjects
    num_modalities = dataset_info['modalities']

    if trainTestFlag == 'test':
        num_sim_train = test_conf['num_dataset']
        iters = range(num_sim_train)
        num_all_data = num_sim_train
    elif trainTestFlag == 'eval':
        num_sim_train = test_conf['num_dataset']

        if contrast_type == 'mc':
            num_sim_test = test_conf['num_test_dataset']
            iters = itertools.product(range(num_sim_train), range(num_sim_test))
        elif contrast_type == 'rc':
             iters = range(num_sim_train)
             num_sim_test = 1 # was test_conf['num_test_dataset']
        else:
            print('Error!')
        num_all_data = num_sim_train*num_sim_test
    else :
        raise ValueError("trainTestFlag should be declared as 'test'/'eval'")

    # for random-contrast: num_sim_test should be 1 !!
    for idx in range(num_test_sbj):
        for idx2 in range(num_modalities):
            # agg_im = np.zeros(mri_dims)
            for idx3, i_dataset in enumerate(iters):
                im_recon_path = read_result_volume_path(gen_conf, test_conf, i_dataset) # load reconstructed image, shape : (subject, modality, mri_shape)
                im_recon = read_volume(im_recon_path[idx*num_modalities+idx2])
                if idx3 == 0:
                    agg_im = np.zeros_like(im_recon)
                if test_conf['ensemble_type'] == 'ave' or test_conf['ensemble_type'] == 'ave|rc':
                    agg_im += im_recon / num_all_data

            # save agg im
            save_volume(gen_conf, test_conf, agg_im, (idx, idx2, test_conf['ensemble_type'])) # save agg im

    for i_dataset in iters:
        ## clean single model outputs
        if isClearSingleOutputs is True:
            clear_result_volume(gen_conf, test_conf, i_dataset)

    return True

def aggregate_all_models(gen_conf, test_conf, train_conf = None, trainTestFlag='eval', isClearSingleOutputs=False, contrast_type='mc'):
    '''
    update:
    1. (23/05/15) Remove mri_dims and load mri dim from volumes input
    '''
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_test_sbj = dataset_info['num_volumes'][1] # number of test subjects
    num_modalities = dataset_info['modalities']

    if trainTestFlag == 'test':
        num_sim_train = test_conf['num_dataset']
        iters = range(num_sim_train)
        num_all_data = num_sim_train
    elif trainTestFlag == 'eval':
        num_sim_train = test_conf['num_dataset']
        num_sim_test = test_conf['num_test_dataset']
        if contrast_type == 'mc':
            iters = itertools.product(range(num_sim_train), range(num_sim_test))
        elif contrast_type == 'rc':
            iters = range(num_sim_train)
        else:
            print("Error!")
        num_all_data = num_sim_train*num_sim_test
    else :
        raise ValueError("trainTestFlag should be declared as 'test'/'eval'")
   #
    # for random-contrast: num_sim_test should be 1 !!
    for idx, i_dataset in enumerate(iters):
        im_recon = read_result_volume(gen_conf, test_conf, i_dataset) # load reconstructed image, shape : (subject, modality, mri_shape)
        if test_conf['ensemble_type'] == 'ave' or test_conf['ensemble_type'] == 'ave|rc':
            if idx == 0:
                agg_im = np.zeros((num_test_sbj, num_modalities,)+im_recon.shape)
            agg_im += im_recon / num_all_data

            ## clean single model outputs
            if isClearSingleOutputs is True:
                clear_result_volume(gen_conf, test_conf, i_dataset)

    # save agg im
    for idx in range(num_test_sbj):
        for idx2 in range(num_modalities):
            save_volume(gen_conf, test_conf, agg_im[idx, idx2], (idx, idx2, test_conf['ensemble_type'])) # save agg im


    return True

def testing_uncertain_model_random_contrast(gen_conf, test_conf, train_conf = None, i_dataset = None, is_lowcost = True) :

    print("Start testing ... load data, mean, std and model...")

    if is_lowcost is True:
        input_data, _ = read_data_path(gen_conf, test_conf, 'test', originProcessFlag="process", i_dataset=i_dataset)
        input_data = np.array(input_data) # input data path
        print(input_data)
    else:
        ## load data
        input_data, _ = read_dataset(gen_conf, test_conf, 'test', i_dataset=i_dataset) # todo: to be developed for ICH dataset

    if train_conf is None:
        conf = test_conf
    else:
        conf = train_conf

    mean, std = read_dataloader_meanstd(gen_conf, conf, 'all') # without norm 13/10
    model = read_model(gen_conf, conf, 'all')

    print("Test model ...")
    test_model_batchensemble(gen_conf, test_conf, input_data, model, mean, std, i_dataset)  # comment cauz' debugging

    return model

def testing_single_model_random_contrast(gen_conf, test_conf, train_conf = None, i_dataset = None, is_lowcost = True) :

    print("Start testing ... load data, mean, std and model...")

    start = time.time()

    if is_lowcost is True:
        input_data, _ = read_data_path(gen_conf, test_conf, 'test', originProcessFlag="process", i_dataset=i_dataset)
        input_data = np.array(input_data) # input data path
        print(input_data)
    else:
        ## load data
        input_data, _ = read_dataset(gen_conf, test_conf, 'test', i_dataset=i_dataset) # todo: to be developed for ICH dataset

    if train_conf is None:
        conf = test_conf
    else:
        conf = train_conf

    mean, std = read_dataloader_meanstd(gen_conf, conf, 'all') # without norm 13/10
    model = read_model(gen_conf, conf, 'all')

    print("Test model ...")
    test_model_3(gen_conf, test_conf, input_data, model, mean, std, i_dataset)  # comment cauz' debugging

    end = time.time()
    etime = end - start
    print("Elapsed time: %.2f" % etime)

    return model

def testing_single_model_multi_contrast(gen_conf, test_conf, train_conf = None, i_dataset = None, is_lowcost = True) :
    dataset_info = gen_conf['dataset_info'][test_conf['dataset']]
    num_modalities = dataset_info['modalities']
    is_selfnormalised = test_conf['is_selfnormalised']

    print("Start testing ... load data, mean, std and model...")
    ## load data
    if is_lowcost is True:
        input_data, _ = read_data_path(gen_conf, test_conf, 'test', originProcessFlag="process", i_dataset=i_dataset)
        input_data = np.array(input_data) # input data path
        print(input_data)
    else:
        input_data, _ = read_dataset(gen_conf, test_conf, 'test', i_dataset=i_dataset)

    if train_conf is None:
        conf = test_conf
    else:
        conf = train_conf

    start = time.time()

    mean, std = read_dataloader_meanstd(gen_conf, conf, 'all') # without norm 13/10
    model = read_model(gen_conf, conf, 'all')

    if is_selfnormalised == True:
        print("Self normalize data ...")
        x_mean, x_std = compute_statistics(input_data, num_modalities)
        input_data = normalise_volume(input_data, num_modalities, x_mean, x_std)

    print("Test model ...")
    test_model_3(gen_conf, test_conf, input_data, model, mean, std, i_dataset)  # comment cauz' debugging
    return model

def testing(gen_conf, test_conf, train_conf = None, i_dataset = None) :
    dataset_info = gen_conf['dataset_info'][test_conf['dataset']]

    num_modalities = dataset_info['modalities']
    is_selfnormalised = test_conf['is_selfnormalised']

    print("Start testing ... load data, mean, std and model...")
    ## load data
    input_data, _ = read_dataset(gen_conf, test_conf, 'test')

    if train_conf is None:
        conf = test_conf
    else:
        conf = train_conf

    mean, std = read_dataloader_meanstd(gen_conf, conf, i_dataset=i_dataset) # without norm 13/10
    model = read_model(gen_conf, conf, i_dataset)

    if is_selfnormalised == True:
        print("Self normalize data ...")
        x_mean, x_std = compute_statistics(input_data, num_modalities)
        input_data = normalise_volume(input_data, num_modalities, x_mean, x_std)

    print("Test model ...")
    test_model_3(gen_conf, test_conf, input_data, model, mean, std, i_dataset)  # comment cauz' debugging

    return model

# latest one
def test_model_batchensemble(gen_conf,
                 test_conf,
                 input_data,
                 model,
                 mean,
                 std,
                 case_name = None) :
    '''
    history:
    10/12 - input_data now can be a list directed to input data paths
    '''
    if case_name is None:
        case_name = 'all'

    dataset_info = gen_conf['dataset_info'][test_conf['dataset']]
    num_volumes = dataset_info['num_volumes'][1]
    sparse_scale = dataset_info['sparse_scale']
    num_modalities = dataset_info['modalities']
    batch_size = test_conf['batch_size']
    is_selfnormalised = test_conf['is_selfnormalised']
    num_ensembles = test_conf['num_ensembles'] # num of ensembles models

    if 'test_batches' not in test_conf:
        test_batches = 1000
    else:
        test_batches = test_conf['test_batches']

    test_indexes = range(0, num_volumes)

    im_recon_path = read_result_volume_path(gen_conf, test_conf, str(case_name) + '{}')  # load reconstructed image, shape : (subject, modality, mri_shape)

    for idx, test_index in enumerate(test_indexes) :

        # 11/21 skip computing the outcomes that are existed, default: only considering modality no.0
        if not (os.path.isfile(im_recon_path[idx*num_modalities].format(''))
                and os.path.isfile(im_recon_path[idx*num_modalities].format('_std')) ):

            ## should only sample one subject
            if input_data.ndim == 6 :
                input_data_temp = np.reshape(input_data[test_index], (1, )+input_data.shape[1:2]+input_data.shape[3:])
                in_volume_shape = input_data_temp.shape[2:]  # bug?
            elif input_data.ndim == 1:
                # input data path instead of data itself, load data (Note: single modality)
                input_data_temp = read_volume(input_data[test_index])
                in_volume_shape = input_data_temp.shape
                input_data_temp = np.reshape(input_data_temp, (1, 1,) + input_data_temp.shape)
            else:
                input_data_temp = np.reshape(input_data[test_index], (1, )+input_data.shape[1:])
                in_volume_shape = input_data_temp.shape[1:4]

            out_volume_shape = np.array(in_volume_shape, dtype=np.int) * np.array(sparse_scale, dtype=np.int)

            x_test, _ = overlap_patching(gen_conf, test_conf, input_data_temp,
                         output_data = None,
                         trainTestFlag = 'test',
                         representative_modality = 0)
            del input_data_temp

            # normalisation before prediction # without norm 13/10
            if is_selfnormalised == False:
                x_test = normalise_volume(x_test, num_modalities, mean['input'], std['input'])

            # new for batchensembles: replicate each patch for n time, n=#ensemble models
            ## if x_test becomes too large after replication, we need to optimise the memory footprint in the future.
            ## 11/16 we have observed the large memory footprint issue, need a way (solved)
            ## so here the recommended #ensembles = 16.
            x_test = np.repeat(x_test, num_ensembles, axis=0)
            x_test_split_indices = [*range(num_ensembles*test_batches, len(x_test), num_ensembles*test_batches)] # divisible by num_ensembles

            # split x_test: split dim should be divisible by num_ensembles !!!
            x_test = np.vsplit(x_test, x_test_split_indices)

            # # version two: save temp memory
            mean_recon_im = []
            std_recon_im = []
            for _ in range(len(x_test)):
                recon_im_temp = model.predict(x_test[0], batch_size=batch_size, verbose=test_conf['verbose'])
                # de-normalisation before prediction
                recon_im_temp = denormalise_volume(recon_im_temp, num_modalities, mean['output'], std['output'])  # without norm 13/10

                recon_im_temp = np.stack(np.vsplit(recon_im_temp, len(recon_im_temp) // num_ensembles))
                mean_recon_im_temp = np.mean(recon_im_temp, axis=1)
                std_recon_im_temp = np.sqrt(np.mean(recon_im_temp ** 2, axis=1) - mean_recon_im_temp ** 2)

                mean_recon_im.append(mean_recon_im_temp)
                std_recon_im.append(std_recon_im_temp)
                x_test.pop(0)  # remove the first x_test element
            mean_recon_im = np.vstack(mean_recon_im)
            std_recon_im = np.vstack(std_recon_im)

            # recon_im = recon_im.reshape((len(recon_im),) + output_shape + (num_classes,))
            for idx2 in range(num_modalities) : # actually num_modalities == 1 so it does nothing
                print("Reconstructing ...")
                # mean
                mean_recon_im2 = reconstruct_volume_imaging(gen_conf, test_conf, mean_recon_im[:, idx2])
                save_volume(gen_conf, test_conf, mean_recon_im2, (idx, idx2, case_name)) # we dont specify 'mean' in the filename.

                #std
                std_recon_im2 = reconstruct_volume_imaging(gen_conf, test_conf, std_recon_im[:, idx2])
                save_volume(gen_conf, test_conf, std_recon_im2, (idx, idx2, case_name, 'std'))

    return True


# latest one
def test_model_3(gen_conf,
                 test_conf,
                 input_data,
                 model,
                 mean,
                 std,
                 case_name = None) :
    '''
    history:
    10/12 - input_data now can be a list directed to input data paths
    '''
    if case_name is None:
        case_name = 'all'

    dataset_info = gen_conf['dataset_info'][test_conf['dataset']]
    sparse_scale = dataset_info['sparse_scale']
    num_volumes = dataset_info['num_volumes'][1]
    num_modalities = dataset_info['modalities']
    batch_size = test_conf['batch_size']
    is_selfnormalised = test_conf['is_selfnormalised']

    test_indexes = range(0, num_volumes)
    for idx, test_index in enumerate(test_indexes) :

        ## should only sample one subject
        if input_data.ndim == 6 :
            input_data_temp = np.reshape(input_data[test_index], (1, )+input_data.shape[1:2]+input_data.shape[3:])
            in_volume_shape = input_data_temp.shape[2:] # bug?
        elif input_data.ndim == 1:
            # input data path instead of data itself, load data
            input_data_temp = read_volume(input_data[test_index])
            in_volume_shape = input_data_temp.shape
            input_data_temp = np.reshape(input_data_temp, (1, 1,) + input_data_temp.shape)
        else:
            input_data_temp = np.reshape(input_data[test_index], (1, )+input_data.shape[1:])
            in_volume_shape = input_data_temp.shape[1:4]

        out_volume_shape = np.array(in_volume_shape, dtype=np.int) * np.array(sparse_scale, dtype=np.int)

        x_test, _ = overlap_patching(gen_conf, test_conf, input_data_temp,
                     output_data = None,
                     trainTestFlag = 'test',
                     representative_modality = 0)

        # normalisation before prediction # without norm 13/10
        if is_selfnormalised == False:
            x_test = normalise_volume(x_test, num_modalities, mean['input'], std['input'])

        recon_im = model.predict(x_test,
                                 batch_size=batch_size,
                                 verbose=test_conf['verbose'])
        # de-normalisation before prediction
        recon_im = denormalise_volume(recon_im, num_modalities, mean['output'], std['output']) # without norm 13/10

        # recon_im = recon_im.reshape((len(recon_im),) + output_shape + (num_classes,))
        for idx2 in range(num_modalities) :
            print("Reconstructing ...")
            # new feature
            # recon_im2 = reconstruct_volume_imaging_with_correction(gen_conf, test_conf, recon_im[:, idx2])
            recon_im2 = reconstruct_volume_imaging(gen_conf, test_conf, recon_im[:, idx2], out_volume_shape)
            # recon_im2 = reconstruct_volume_imaging3(gen_conf, test_conf, recon_im[:, idx2])
            save_volume(gen_conf, test_conf, recon_im2, (idx, idx2, case_name))
        del x_test, recon_im

    return True

def normalise_volume(input_data, num_modalities, mean, std) :
    input_data_tmp = np.copy(input_data)
    for vol_idx in range(len(input_data_tmp)) :
        for modality in range(num_modalities) :
            input_data_tmp[vol_idx, modality] -= mean[modality]
            input_data_tmp[vol_idx, modality] /= std[modality]
    return input_data_tmp

def denormalise_volume(input_data, num_modalities, mean, std) :
    input_data_tmp = np.copy(input_data)
    for vol_idx in range(len(input_data_tmp)) :
        for modality in range(num_modalities) :
            input_data_tmp[vol_idx, modality] *= std[modality]
            input_data_tmp[vol_idx, modality] += mean[modality]
    return input_data_tmp

def compute_statistics(input_data, num_modalities) :
    mean = np.zeros((num_modalities, ))
    std = np.zeros((num_modalities, ))

    for modality in range(num_modalities) :
        modality_data = input_data[:, modality]
        mean[modality] = np.mean(modality_data)
        std[modality] = np.std(modality_data)

    return mean, std
