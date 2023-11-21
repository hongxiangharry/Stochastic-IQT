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
# This code was dervied from early version of intensity normalization package: https://github.com/jcreinhold/intensity-normalization
# @Citation: Reinhold, Jacob C., et al. "Evaluating the impact of intensity normalization on MR image synthesis."
# Medical Imaging 2019: Image Processing. Vol. 10949. SPIE, 2019.
#
"""  """

from intensity_normalization.normalize.nyul import train, do_hist_norm
import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import os
import numpy as np
from utils.ioutils import read_data_path
import itertools

def separate_normalisation(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag = 'train', out_suffix = '_norm'):
    if 'contrast_type' not in sim_norm_conf['sim'].keys():
        print('Use contrast type as mc ..')
        sim_norm_conf['sim']['contrast_type'] = 'mc'

    if sim_norm_conf['sim']['contrast_type'] == 'mc' or sim_norm_conf['sim']['contrast_type'] == 'multi-contrast':
        print('Run normalization on MC ...')
        separate_normalisation_with_multi_contrast(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag, out_suffix)
    elif sim_norm_conf['sim']['contrast_type'] == 'rc' or sim_norm_conf['sim']['contrast_type'] == 'random-contrast':
        print('Run normalization on RC ...')
        separate_normalisation_with_random_contrast(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag, out_suffix)
    else:
        raise Exception("contrast type has not been defined ...")

    return True


def separate_normalisation_with_multi_contrast(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag = 'train', out_suffix = '_norm'):
    '''
    Train: 1. hist train, norm and thresholding img, 2. same to gt
    Eval: 1. hist load, norm and thresholding img, 2. same to gt
    Test: 1. hist load, norm and reshape img
    '''
    dataset = traintest_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]

    basepath = os.path.join(gen_conf['dataset_path'], dataset_info['path'][0])  # process dir: '/cluster/project0/IQT_Nigeria/HCP/process/pilot/t1'

    # sim-norm conf
    norm_conf = sim_norm_conf['norm']
    hist_suffixes, hist_pattern = norm_conf['hist_suffixes'], norm_conf['hist_pattern']

    if trainTestFlag in ['train', 'eval']:
        if trainTestFlag == 'train':
            num_sim_train = traintest_conf['num_dataset']
            iters = range(num_sim_train)
        elif trainTestFlag == 'eval':
            num_sim_train = traintest_conf['num_dataset']
            num_sim_test = traintest_conf['num_test_dataset']
            iters = itertools.product(range(num_sim_train), range(num_sim_test))

        for i_dataset in iters:
            # lf
            lf_process_paths, hf_process_paths = read_data_path(gen_conf, traintest_conf, trainTestFlag, originProcessFlag='process', i_dataset=i_dataset)
            lf_hist_fp = generate_hist_path(basepath, [i_dataset, hist_suffixes[0]], hist_pattern=hist_pattern)
            normalise_data(lf_process_paths, lf_hist_fp, trainTestFlag=trainTestFlag, out_suffix=out_suffix)
        # hf
        hf_hist_fp = generate_hist_path(basepath, ['', hist_suffixes[1]], hist_pattern=hist_pattern)
        normalise_data(hf_process_paths, hf_hist_fp, trainTestFlag=trainTestFlag, out_suffix=out_suffix)

    elif trainTestFlag == 'test':
        num_sim_train = traintest_conf['num_dataset']

        reshape_options = {
            'voxel_sizes': dataset_info['sparse_scale'],
            'target_dim': dataset_info['shrink_dim']-1,
            'new_shape': dataset_info['dimensions']
        }
        lf_org_paths, _ = read_data_path(gen_conf, traintest_conf, trainTestFlag = 'test', originProcessFlag = 'origin', i_dataset = None)
        for i_dataset in range(num_sim_train):
            lf_process_paths, _ = read_data_path(gen_conf, traintest_conf, trainTestFlag = 'test', originProcessFlag = 'process', i_dataset = i_dataset)
            lf_hist_fp = generate_hist_path(basepath, [i_dataset, hist_suffixes[0]], hist_pattern = hist_pattern)
            normalise_data(lf_process_paths, lf_hist_fp, in_img_fps = lf_org_paths, trainTestFlag = trainTestFlag, out_suffix = out_suffix, test_options=reshape_options)

    return True

def separate_normalisation_with_random_contrast(gen_conf, train_conf, sim_norm_conf, trainTestFlag = 'train',
                                                test_conf = None, out_suffix = '_norm', is_resampling = False,
                                                is_padding = False, is_background_masking = True):
    '''
    Train: 1. hist train, norm and thresholding img, 2. same to gt
    Eval: 1. hist load, norm and thresholding img, 2. same to gt
    Test: 1. hist load, norm and reshape img
    '''
    # training info
    train_dataset = train_conf['dataset']
    train_dataset_info = gen_conf['dataset_info'][train_dataset]

    basepath = os.path.join(gen_conf['dataset_path'], train_dataset_info['path'][0])

    # sim-norm conf
    sim_conf = sim_norm_conf['sim']
    norm_conf = sim_norm_conf['norm']
    hist_suffixes, hist_pattern = norm_conf['hist_suffixes'], norm_conf['hist_pattern']

    if test_conf is None:
        test_conf = train_conf

    if trainTestFlag == 'train':
        # for training
        num_volumes = train_dataset_info['num_volumes'][0]
        num_sim_train = train_conf['num_dataset'] # says 4 contrasts
        iters = range(num_sim_train)
        if 'num_train_sbj' in train_conf.keys():
            num_volumes = train_conf['num_train_sbj']  # says 60
        volume_indices = np.arange(num_volumes)
        hf_process_paths = []

        for id_contrast in iters:

            contrast_indices = np.array( sim_conf['contrast_indices'] ) # bug cleared
            selected_volume_indices = volume_indices[contrast_indices == id_contrast]
            lf_process_paths = []

            for svid in selected_volume_indices:

                # lf/hf process paths -> list with one elem
                lf_process_path, hf_process_path = read_data_path(gen_conf, train_conf, trainTestFlag, originProcessFlag='process', i_dataset=svid, indices=[svid, 0])
                lf_process_paths.append(lf_process_path[0])
                hf_process_paths.append(hf_process_path[0])

            # lf
            lf_hist_fp = generate_hist_path(basepath, [id_contrast, hist_suffixes[0]], hist_pattern=hist_pattern)
            normalise_data(lf_process_paths, lf_hist_fp, trainTestFlag=trainTestFlag, out_suffix=out_suffix)

        # hf, is it necessary to have? - we can have an experiment! Next work!
        hf_hist_fp = generate_hist_path(basepath, ['', hist_suffixes[1]], hist_pattern=hist_pattern)
        normalise_data(hf_process_paths, hf_hist_fp, trainTestFlag=trainTestFlag, out_suffix=out_suffix)

    elif trainTestFlag == 'eval':
        # for validation
        test_dataset = test_conf['dataset']
        test_dataset_info = gen_conf['dataset_info'][test_dataset]

        num_volumes = test_dataset_info['num_volumes'][1]
        num_sim_train = train_conf['num_dataset'] # it should be same to train_dataset_info['num_volumes'][0]
        iters = itertools.product(range(num_sim_train), range(num_volumes))

        for i_dataset in iters:
            if isinstance(i_dataset, tuple):
                idx = i_dataset[1]  # eval, test sbj
            else:
                idx = i_dataset # train sbj
            # lf
            lf_process_paths, hf_process_paths = read_data_path(gen_conf, test_conf, trainTestFlag, originProcessFlag='process', i_dataset=i_dataset, indices=[idx, 0])
            lf_hist_fp = generate_hist_path(basepath, [i_dataset, hist_suffixes[0]], hist_pattern=hist_pattern)
            normalise_data(lf_process_paths, lf_hist_fp, trainTestFlag=trainTestFlag, out_suffix=out_suffix)
            if i_dataset[0] == 0:
                # hf
                hf_hist_fp = generate_hist_path(basepath, ['', hist_suffixes[1]], hist_pattern=hist_pattern)
                normalise_data(hf_process_paths, hf_hist_fp, trainTestFlag=trainTestFlag, out_suffix=out_suffix)

    elif trainTestFlag == 'test':
        # for test
        test_dataset = test_conf['dataset']
        test_dataset_info = gen_conf['dataset_info'][test_dataset]

        if 'dimensions' in test_dataset_info.keys():
            new_shape = np.array(test_dataset_info['dimensions']) // np.array(test_dataset_info['sparse_scale'])
        else:
            new_shape = None
        test_options = {
            'voxel_sizes': test_dataset_info['sparse_scale'] if is_resampling is True else None,
            'target_dim': test_dataset_info['shrink_dim']-1,
            'new_shape': new_shape if is_padding is True else None,
            'is_background_masking': is_background_masking
        }

        num_volumes = test_dataset_info['num_volumes'][1] # test_conf
        num_sim_train = train_conf['num_dataset']
        iters = itertools.product(range(num_sim_train), range(num_volumes))

        for i_dataset in iters:
            if isinstance(i_dataset, tuple):
                idx = i_dataset[1]  # eval, test sbj
            else:
                idx = i_dataset # train sbj
            lf_org_paths, _ = read_data_path(gen_conf, test_conf, trainTestFlag='test', originProcessFlag='origin',
                                                 i_dataset=i_dataset, indices=[idx, 0])
            lf_process_paths, _ = read_data_path(gen_conf, test_conf, trainTestFlag = 'test', originProcessFlag = 'process',
                                                 i_dataset=i_dataset, indices=[idx, 0])
            lf_hist_fp = generate_hist_path(basepath, [i_dataset, hist_suffixes[0]], hist_pattern = hist_pattern)
            normalise_data(lf_process_paths, lf_hist_fp, in_img_fps = lf_org_paths, trainTestFlag = trainTestFlag,
                           out_suffix = out_suffix, test_options=test_options)

    return True


def merged_normalisation_random_contrast(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag = 'train',
                                         train_conf=None, out_suffix = '_norm', is_resampling = False,
                                         is_padding = False, is_background_masking = True):
    '''
    Normalise all simulation with the same hist
    one train model - possible to reuse separate_normalisation_with_random_contrast
    if trainTestFlag = 'train' traintest_conf for train, train_conf = None -> traintest_conf
    if trainTestFlag = 'eval' and 'test' traintest_conf for test, train_conf is not None

    '''

    if trainTestFlag == 'train':
        train_conf = traintest_conf
        test_conf = None
    elif trainTestFlag in ['eval', 'test']:
        if train_conf is None:
            train_conf = traintest_conf
        test_conf = traintest_conf

    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]

    if 'num_train_sbj' in train_conf.keys():
        num_volumes = train_conf['num_train_sbj']
    else:
        num_volumes = dataset_info['num_volumes'][0]
    train_conf['num_dataset'] = 1 # in case num_dataset is not 1.

    sim_norm_conf['sim']['contrast_indices'] = [0,] * num_volumes # adapt to merge mode
    separate_normalisation_with_random_contrast(gen_conf, train_conf, sim_norm_conf, trainTestFlag=trainTestFlag,
                                                test_conf = test_conf, out_suffix=out_suffix,
                                                is_resampling=is_resampling, is_padding=is_padding,
                                                is_background_masking=is_background_masking)
    return True

def generate_hist_path(basepath, types, hist_pattern = 'hist_{}_{}.npy'):
    '''
    generate histogram path to a place for both train and test
    :param basepath:
    :param types: 0->i_dataset, 1-> lf/hf
    :param hist_pattern:
    :return:
    '''
    if isinstance(types[0], tuple):
        types[0] = types[0][0]

    hist_path = os.path.join(basepath, '..', hist_pattern).format(types[0], types[1])
    return hist_path

def normalise_data(img_fps, hist_fp, in_img_fps = None, trainTestFlag ='train', out_suffix = None, test_options = None):
    ''' histogram normalization on input and gt data for HCP-t1
        train: train & save hist, applying to img and gt
        eval: load hist, applying to img and gt
        test: load hist, applying to img
            voxel_sizes = [1, 1, 8], target_dim = 2, new_shape = [465, 465, 27]
        img_fps: (all mode) image file paths for training histogram landmark
        hist_fp: file path of histogram landmark
        in_img_fps: 'test' mode only. It will be the inputs, img_fps will be the outputs.
        # 23/06/06 update:
        test_option:
        - padding
        - reshape
        - apply mask to eliminate non-zero background
    '''

    if in_img_fps is None:
        in_img_fps = img_fps

    ## input
    if trainTestFlag == 'train':
        standard_scale, percs = train(img_fps, mask_fns=None, i_min=1, i_max=99, i_s_min=1, i_s_max=100, l_percentile=10, u_percentile=90, step=10)
        if not os.path.isdir(os.path.dirname(hist_fp)):
            os.makedirs(os.path.dirname(hist_fp))
        np.save(hist_fp, np.vstack((standard_scale, percs)))
    else:
        standard_scale, percs = np.load(hist_fp)

    # apply histogram landmark to data
    # if out_suffix == None, input: fp, output: fp
    # if out_suffix is not None, input: out_fp, output: out_fp
    for fp, in_fp in zip(img_fps, in_img_fps):
        # img
        print("Matching histogram of image: ", fp)
        if out_suffix is not None:
            out_base, out_ext = split_nii_path(fp)
            out_fp = out_base + out_suffix + out_ext
        else:
            out_fp = fp

        if not os.path.isdir(os.path.dirname(out_fp)):
            os.makedirs(os.path.dirname(out_fp))

        img = nib.load(in_fp)
        img = do_hist_norm(img, percs, standard_scale, mask=None)

        # # apply mask
        # img = nib.Nifti1Image( img.get_fdata()*mask.get_fdata(), img.affine)
        nib.save(img, out_fp)

    ## 0-thresholding training data and gt
    # if out_suffix == None, input: fp, output: fp
    # if out_suffix is not None, input: out_fp, output: out_fp
    if trainTestFlag in ['train', 'eval']:
        for fp in img_fps:
            # img
            print("Thresholding image: ", fp)
            if out_suffix is not None:
                out_base, out_ext = split_nii_path(fp)
                out_fp = out_base + out_suffix + out_ext
                fp = out_fp
            else:
                out_fp = fp
            if not os.path.isdir(os.path.dirname(out_fp)):
                os.makedirs(os.path.dirname(out_fp))

            img = nib.load(fp)
            img = nib.Nifti1Image(img.get_fdata() * (img.get_fdata() >= 0), img.affine) # may lost some csf pixels, but this is the easiest way to set background as 0 without shifting histggram (iqt-21)
            nib.save(img, out_fp)
    elif trainTestFlag == 'test':
        if 'voxel_sizes' in test_options.keys():
            voxel_sizes = test_options['voxel_sizes'] # 'sparse_scale'
        else:
            voxel_sizes = None
        target_dim = test_options['target_dim'] # dataset_info['shrink_dim']-1, only support single dimension now
        if 'new_shape' in test_options.keys():
            new_shape = test_options['new_shape'] # for padding, dataset_info['dimensions'] // 'sparse_scale'
        else:
            new_shape = None
        if 'is_background_masking' in test_options.keys():
            is_background_masking = test_options['is_background_masking']
        else:
            is_background_masking = False

        for fp, in_fp in zip(img_fps, in_img_fps):
            # img
            print("Post-processing test image: ", fp)
            if out_suffix is not None:
                out_base, out_ext = split_nii_path(fp)
                out_fp = out_base + out_suffix + out_ext
                fp = out_fp
            else:
                out_fp = fp
            if not os.path.isdir(os.path.dirname(out_fp)):
                os.makedirs(os.path.dirname(out_fp))

            mri_handle = nib.load(filename=fp)
            if is_background_masking is True:
                ## 1. Extract background values from the eight corner voxels
                ## 2. Most frequent value will be the background value

                print("Run masking background sub-step ...")
                in_data = nib.load(in_fp).get_fdata()
                eight_corner_values = [
                    in_data[0,0,0], in_data[0, 0, -1], in_data[0, -1, 0], in_data[0, -1, -1],
                    in_data[-1, 0, 0], in_data[-1, 0, -1], in_data[-1, -1, 0], in_data[-1, -1, -1],
                ]
                eight_corner_counts = {i:eight_corner_values.count(i) for i in set(eight_corner_values)}
                bg_value = max(zip(eight_corner_counts.values(), eight_corner_counts.keys()))[1]

                mri_volume = mri_handle.get_fdata() * (in_data != bg_value)
                mri_handle = nib.Nifti1Image(mri_volume, mri_handle.affine)

            # resampling
            if voxel_sizes is not None:
                print("Run resampling sub-step ... ")
                spacing = np.array(mri_handle.header['pixdim'][1:4])
                target_spacing = np.array(voxel_sizes) / voxel_sizes[target_dim] * spacing[target_dim]

                scales = target_spacing / spacing
                i_affine = np.dot(mri_handle.affine, np.diag(np.concatenate((scales, [1]))))  # affine rescaling
                i_shape = np.ceil(np.array(mri_handle.shape) / scales).astype(int)

                mri_handle = resample_from_to(mri_handle, (i_shape, i_affine), order=3)

            # pad image
            if new_shape is not None:
                print("Run padding sub-step ... ")
                mri_data = mri_handle.get_fdata()
                mri_volume = np.zeros(new_shape)
                mri_volume[:mri_handle.shape[0], :mri_handle.shape[1], :mri_handle.shape[2]] = mri_data * (mri_data > 0)
                mri_handle = nib.Nifti1Image(mri_volume, mri_handle.affine)

            nib.save(mri_handle, out_fp)

    return True

def split_nii_path(fn):
    ''' split a filepath into path and extension (works with .nii.gz) '''
    out_base, out_ext = os.path.splitext(fn)
    if out_ext == '.gz':
        out_base, out_ext2 = os.path.splitext(out_base)
        out_ext =  out_ext2 + out_ext

    return out_base, out_ext