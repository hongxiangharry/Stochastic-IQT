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
""" Utility for building patch library."""

import nibabel as nib
import numpy as np
import os
from utils.ioutils import save_dataloader_meanstd, read_data_path
from utils.patching_utils import overlap_patching
import zipfile
import math

def build_patch_lib(gen_conf, train_conf, i_dataset=None) :
    dataset = train_conf['dataset']
    if dataset in ['HCP', 'HCP-Wu-Minn-Contrast-Multimodal', 'HCP-rc', 'MBB', 'MBB-rc']:
        print(dataset)
        return build_general_patchset(gen_conf, train_conf, i_dataset)

def build_patch_lib_random_contrasts(gen_conf, train_conf, i_dataset=None):
    dataset = train_conf['dataset']
    if dataset in ['HCP', 'HCP-Wu-Minn-Contrast-Multimodal', 'HCP-rc', 'MBB', 'MBB-rc']:
        print(dataset)
        return build_general_patchset_random_contrasts(gen_conf, train_conf, i_dataset)

def outlier_detection_indices_loader(path, n_subs, n_modalities):
    od_indices = np.loadtxt(path)
    od_indices = np.reshape(od_indices, (n_subs, n_modalities))
    return od_indices

def build_general_patchset_random_contrasts(gen_conf, train_conf, i_dataset=None):
    '''
    update:
    1. remove dimension of MRI for enabling random shape
    '''
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    patch_shape = train_conf['patch_shape'] # input patch shape
    output_patch_shape = train_conf['output_shape'] # output patch shape
    extraction_step = train_conf['extraction_step'] # step-size for patch extraction step
    mnppf = train_conf['max_num_patches_per_file'] #

    modalities = dataset_info['modalities']
    out_path = dataset_info['path'][3] # patch restored in the project storage

    in_postfixes = dataset_info['in_postfix']

    max_num_patches_per_vol = train_conf['max_num_patches']

    subject_lib = dataset_info['training_subjects']
    num_volumes = train_conf['num_train_sbj'] # actual num of train sbj used to, 60

    if i_dataset is None:
        i_dataset = 'all'

    if i_dataset is None or i_dataset == 'all':
        in_postfixes_arr = in_postfixes
        img_id_arr = range(num_volumes)
    else:
        in_postfixes_arr = [in_postfixes[i_dataset]]
        img_id_arr = [i_dataset]

    rebuild = train_conf['patchlib-rebuild']

    th_info = train_conf['th_info']  # threshold info

    out_patch_dir = os.path.join(dataset_path, out_path)
    if not os.path.isdir(out_patch_dir):
        os.makedirs(out_patch_dir)

    patch_filename_pattern = train_conf['patch_filename_pattern']

    out_patches_filename = patch_filename_pattern.format(i_dataset, patch_shape, extraction_step, num_volumes, 'zip')

    os.chdir(out_patch_dir)

    in_mean_arr = np.zeros((1,))
    in_var_arr = np.zeros((1,))
    out_mean_arr = np.zeros((1,))
    out_var_arr = np.zeros((1,))

    if ( os.path.isfile(out_patches_filename) == False) or ( rebuild == True ) :
        count_patches = 0
        with zipfile.ZipFile(out_patches_filename, 'w') as z:
            for (img_idx, in_postfix) in zip(img_id_arr, in_postfixes_arr):
                # load source files
                for mod_idx in range(modalities): # by default, 1
                    in_filepaths, out_filepaths = read_data_path(gen_conf, train_conf, trainTestFlag='train',
                                   originProcessFlag='process', i_dataset=img_idx, indices=(img_idx, mod_idx))
                    print('Print LF and HF data paths: ', in_filepaths, out_filepaths)

                    in_data = read_volume(in_filepaths[0]).astype(np.float32)
                    in_data = np.expand_dims(np.expand_dims(in_data, axis=0 ), axis=0)
                    # print(in_data.shape)

                    out_data = read_volume(out_filepaths[0]).astype(np.float32)
                    out_data = np.expand_dims(np.expand_dims(out_data, axis=0 ), axis=0)
                    # print(out_data.shape)

                    # build patch lib, dim =
                    in_patches, out_patches = overlap_patching(gen_conf, train_conf, in_data, out_data, th_info = th_info)
                    # randomly some patches at most, update[31/08/20]
                    if max_num_patches_per_vol < len(in_patches):
                        random_order = np.random.choice(len(in_patches), max_num_patches_per_vol, replace=False)
                        in_patches = in_patches[random_order]
                        out_patches = out_patches[random_order]

                    count_patches += len(in_patches)
                    print("Extracted {} patches in this volume, allow {} patches."
                          .format(len(in_patches), max_num_patches_per_vol))
                    patch_filename_template = generate_patch_filename( img_idx, subject_lib[img_idx], patch_shape, extraction_step, "{:04d}") # contrast_id -> img_idx, back to origin

                    for sub_idx in range(math.ceil(len(in_patches) / mnppf)):
                        in_patch = in_patches[sub_idx*mnppf : np.minimum((sub_idx+1)*mnppf, len(in_patches))]
                        out_patch = out_patches[sub_idx*mnppf : np.minimum((sub_idx+1)*mnppf, len(out_patches))]
                        patch_filename = patch_filename_template.format(sub_idx)
                        save_patch_data(in_patch, out_patch, patch_filename)
                        print("Zipping " + patch_filename + " ...")
                        z.write(patch_filename)
                        os.remove(patch_filename)

                    # calculate sum and sum of square
                    in_mean_arr  = in_mean_arr  + np.sum(in_patches, axis=(0, 2, 3, 4))
                    out_mean_arr = out_mean_arr + np.sum(out_patches, axis=(0, 2, 3, 4))
                    in_var_arr   = in_var_arr   + np.sum(in_patches**2, axis=(0, 2, 3, 4))
                    out_var_arr  = out_var_arr  + np.sum(out_patches**2, axis=(0, 2, 3, 4))

        in_mean_arr  /= count_patches*np.prod(patch_shape)
        in_var_arr   /= count_patches*np.prod(patch_shape)
        in_var_arr   =  (in_var_arr - in_mean_arr**2) ** (1 / 2)
        out_mean_arr /= count_patches * np.prod(output_patch_shape)
        out_var_arr  /= count_patches * np.prod(output_patch_shape)
        out_var_arr  =  (out_var_arr - out_mean_arr ** 2) ** (1 / 2)

        mean = {'input': in_mean_arr, 'output': out_mean_arr}
        std = {'input': in_var_arr, 'output': out_var_arr}
        ## save mean and std
        save_dataloader_meanstd(gen_conf, train_conf, mean, std, i_dataset)
    else:
        print("The file '{}' is existed... ".format(out_patches_filename))
        count_patches = None

    if 'job_id' in gen_conf.keys(): # unzip the patch lib
        unzip_dir = dataset_info['path'][2]
        print("Unzip data to "+unzip_dir+' ...')
        if not os.path.isdir(unzip_dir):
            os.makedirs(unzip_dir)
        with zipfile.ZipFile(out_patches_filename) as z:
            z.extractall(path=unzip_dir)

    return count_patches

def build_general_patchset(gen_conf, train_conf, i_dataset=None):
    '''
    update:
    1. multiple contrast case: in_postfix and image are independent, the product forms a complete simulation
    2. remove dimension of MRI for enabling random shape
    '''
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    patch_shape = train_conf['patch_shape'] # input patch shape
    output_patch_shape = train_conf['output_shape'] # output patch shape
    extraction_step = train_conf['extraction_step'] # step-size for patch extraction step
    mnppf = train_conf['max_num_patches_per_file'] #

    modalities = dataset_info['modalities']
    out_path = dataset_info['path'][3] # patch restored in the project storage

    in_postfixes = dataset_info['in_postfix']
    if i_dataset is None or i_dataset == 'all':
        in_postfix_idx_arr = range(len(in_postfixes))
    else:
        in_postfix_idx_arr = [i_dataset]

    max_num_patches_per_vol = train_conf['max_num_patches']

    if i_dataset is None:
        i_dataset = 'all'

    subject_lib = dataset_info['training_subjects']
    num_volumes = train_conf['num_train_sbj'] # actual num of train sbj used to, 15

    rebuild = train_conf['rebuild']

    th_info = train_conf['th_info']  # threshold info

    out_patch_dir = os.path.join(dataset_path, out_path)
    if not os.path.isdir(out_patch_dir):
        os.makedirs(out_patch_dir)

    patch_filename_pattern = train_conf['patch_filename_pattern']

    out_patches_filename = patch_filename_pattern.format(i_dataset, patch_shape, extraction_step, num_volumes, 'zip')

    os.chdir(out_patch_dir)

    in_mean_arr = np.zeros((1,))
    in_var_arr = np.zeros((1,))
    out_mean_arr = np.zeros((1,))
    out_var_arr = np.zeros((1,))

    if not os.path.isfile(out_patches_filename) or rebuild is True:
        count_patches = 0
        with zipfile.ZipFile(out_patches_filename, 'w') as z:
            for img_idx in range(num_volumes):
                # load source files
                for mod_idx in range(modalities): # by default, 1
                    for ip_idx in in_postfix_idx_arr:
                        in_filepath, out_filepath = read_data_path(gen_conf, train_conf, trainTestFlag='train',
                                            originProcessFlag='process', i_dataset=ip_idx, indices=(img_idx, mod_idx))
                        print('Print LF and HF data paths: ', in_filepath, out_filepath)
                        in_data = read_volume(in_filepath).astype(np.float32)
                        in_data = np.expand_dims(np.expand_dims(in_data, axis=0 ), axis=0)
                        # print(in_data.shape)

                        out_data = read_volume(out_filepath).astype(np.float32)
                        out_data = np.expand_dims(np.expand_dims(out_data, axis=0 ), axis=0)
                        # print(out_data.shape)

                        # build patch lib, dim = (#
                        in_patches, out_patches = overlap_patching(gen_conf, train_conf, in_data, out_data, th_info = th_info)
                        # randomly some patches at most, update[31/08/20]
                        print("Extracted {} patches in this volume, allow {} patches."
                              .format(len(in_patches), max_num_patches_per_vol))

                        if max_num_patches_per_vol < len(in_patches):
                            random_order = np.random.choice(len(in_patches), max_num_patches_per_vol, replace=False)
                            in_patches = in_patches[random_order]
                            out_patches = out_patches[random_order]

                        count_patches += len(in_patches)
                        print("Extracted {} patches in this volume, allow {} patches, totally cumulative patches = {}."
                              .format(len(in_patches), max_num_patches_per_vol, count_patches))
                        patch_filename_template = generate_patch_filename( ip_idx, subject_lib[img_idx], patch_shape, extraction_step, "{:04d}") # '{}-{}-{}-{}-{}.{}'

                        for sub_idx in range(math.ceil(len(in_patches) / mnppf)):
                            in_patch = in_patches[sub_idx*mnppf:np.minimum((sub_idx+1)*mnppf, len(in_patches))]
                            out_patch = out_patches[sub_idx*mnppf:np.minimum((sub_idx+1)*mnppf, len(out_patches))]
                            patch_filename = patch_filename_template.format(sub_idx)
                            save_patch_data(in_patch, out_patch, patch_filename)
                            print("Zipping " + patch_filename + " ...")
                            z.write(patch_filename)
                            os.remove(patch_filename)

                        # calculate sum and sum of square
                        in_mean_arr = in_mean_arr + np.sum(in_patches, axis=(0, 2, 3, 4))
                        out_mean_arr = out_mean_arr + np.sum(out_patches, axis=(0, 2, 3, 4))
                        in_var_arr = in_var_arr + np.sum(in_patches**2, axis=(0, 2, 3, 4))
                        out_var_arr = out_var_arr + np.sum(out_patches**2, axis=(0, 2, 3, 4))

        in_mean_arr  /= count_patches*np.prod(patch_shape)
        in_var_arr   /= count_patches*np.prod(patch_shape)
        in_var_arr   =  (in_var_arr - in_mean_arr**2) ** (1 / 2)
        out_mean_arr /= count_patches * np.prod(output_patch_shape)
        out_var_arr  /= count_patches * np.prod(output_patch_shape)
        out_var_arr  =  (out_var_arr - out_mean_arr ** 2) ** (1 / 2)

        mean = {'input': in_mean_arr, 'output': out_mean_arr}
        std = {'input': in_var_arr, 'output': out_var_arr}
        ## save mean and std
        save_dataloader_meanstd(gen_conf, train_conf, mean, std, i_dataset)
    else:
        print("The file '{}' is existed... ".format(out_patches_filename))
        count_patches = None

    if 'job_id' in gen_conf.keys(): # unzip the patch lib
        unzip_dir = dataset_info['path'][2]
        print("Unzip data to "+unzip_dir+' ...')
        if not os.path.isdir(unzip_dir):
            os.makedirs(unzip_dir)
        with zipfile.ZipFile(out_patches_filename) as z:
            z.extractall(path=unzip_dir)

    return count_patches

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

def generate_patch_filename( modality, sample_num, patch_shape, extraction_step, sub, extension = 'npz') :
    file_pattern = '{}-{}-{}-{}-{}.{}'
    print(file_pattern.format( modality, sample_num, patch_shape, extraction_step, sub, extension))
    return file_pattern.format( modality, sample_num, patch_shape, extraction_step, sub, extension)

def unzip(src_filename, dest_dir):
    with zipfile.ZipFile(src_filename) as z:
        z.extractall(path=dest_dir)

def compute_statistics(input_data, num_modalities) :
    mean = np.zeros((num_modalities, ))
    std = np.zeros((num_modalities, ))

    for modality in range(num_modalities) :
        modality_data = input_data[:, modality]
        mean[modality] = np.mean(modality_data)
        std[modality] = np.std(modality_data)

    return mean, std