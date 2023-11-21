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
""" Define training data loader for the HCP dataset."""


from tensorflow.keras.utils import Sequence
import numpy as np
import os
import zipfile
import glob
import math
import shutil


def DefineTrainValHCPDataloader(gen_conf, train_conf, i_dataset = None, is_shuffle_trainval = False):
    '''
    - Every patch pack has one patch data inside.
    -
    '''
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    validation_split_ratio = train_conf['validation_split']

    batch_size = train_conf['batch_size']

    path = dataset_info['path'][2]  # patch
    patch_dir = os.path.join(dataset_path, path)
    if i_dataset is None or i_dataset == 'all':
        search_path = patch_dir + '/*.npz'
    else:
        search_path = patch_dir +'/{}-*.npz'.format(i_dataset)

    N_patch_packs = len(sorted(glob.glob(search_path))) # number of patches in the patch lib
    if is_shuffle_trainval is True:
        shuffle_order = np.random.permutation(N_patch_packs)
    else:
        shuffle_order = np.arange(N_patch_packs)

    val_patch_packs = np.int32(np.ceil(N_patch_packs * validation_split_ratio))
    train_patch_packs = N_patch_packs - val_patch_packs

    # divisible by batch_size, updating at 11/16
    train_patch_packs = train_patch_packs // batch_size * batch_size
    val_patch_packs = val_patch_packs // batch_size * batch_size

    trainDataloader = HCPSequence(gen_conf, train_conf, shuffle_order[:train_patch_packs],
                                  train_patch_packs, i_dataset=i_dataset)
    valDataloader = HCPSequence(gen_conf, train_conf, shuffle_order[train_patch_packs:train_patch_packs+val_patch_packs],
                                val_patch_packs, i_dataset=i_dataset)

    return trainDataloader, valDataloader

def read_dataloader_meanstd(gen_conf, train_conf, i_dataset=None):
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    patch_shape = train_conf['patch_shape'] # input patch shape
    extraction_step = train_conf['extraction_step'] # step-size for patch extraction step

    out_path = dataset_info['path'][3] # patch restored in the project storage

    num_volumes = train_conf['num_train_sbj']

    if i_dataset is None:
        i_dataset = 'all'

    out_patch_dir = os.path.join(dataset_path, out_path)

    patch_filename_pattern = train_conf['patch_filename_pattern']
    mean_filename = os.path.join(out_patch_dir, patch_filename_pattern).format(i_dataset, patch_shape, extraction_step, str(num_volumes)+'_mean', 'npz')
    std_filename = os.path.join(out_patch_dir, patch_filename_pattern).format(i_dataset, patch_shape, extraction_step, str(num_volumes)+'_std', 'npz')

    mean = {}
    mean_f = np.load(mean_filename)
    mean['input'] = mean_f['mean_input']
    mean['output'] = mean_f['mean_output']

    std = {}
    std_f = np.load(std_filename)
    std['input'] = std_f['std_input']
    std['output'] = std_f['std_output']
    return mean, std

def save_dataloader_meanstd(gen_conf, train_conf, mean, std, i_dataset=None):
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    patch_shape = train_conf['patch_shape'] # input patch shape
    extraction_step = train_conf['extraction_step'] # step-size for patch extraction step

    out_path = dataset_info['path'][3] # patch restored in the project storage

    num_volumes = train_conf['num_train_sbj']

    if i_dataset is None:
        i_dataset = 'all'

    out_patch_dir = os.path.join(dataset_path, out_path)

    if not os.path.isdir(out_patch_dir):
        os.makedirs(out_patch_dir)

    patch_filename_pattern = train_conf['patch_filename_pattern']
    mean_filename = os.path.join(out_patch_dir, patch_filename_pattern).format(i_dataset, patch_shape, extraction_step, str(num_volumes)+'_mean', 'npz')
    std_filename = os.path.join(out_patch_dir, patch_filename_pattern).format(i_dataset, patch_shape, extraction_step, str(num_volumes)+'_std', 'npz')

    if (mean is None) or (std is None):
        mean = {'input': np.array([0.0]), 'output': np.array([0.0])}
        std = {'input': np.array([1.0]), 'output': np.array([1.0])}
    np.savez(mean_filename, mean_input=mean['input'], mean_output=mean['output'])
    np.savez(std_filename, std_input=std['input'], std_output=std['output'])
    return True

class HCPSequence(Sequence):
    def __init__(self, gen_conf, train_conf, shuffle_order, N, i_dataset = None):
        dataset = train_conf['dataset']
        dataset_path = gen_conf['dataset_path']
        dataset_info = gen_conf['dataset_info'][dataset]

        self.modalities = dataset_info['modalities']

        self.batch_size = train_conf['batch_size']
        self.N = N
        validation_split = train_conf['validation_split']

        # It is an update for local storage setup. All required data have been completely unzipped to the local storage. Need to set up dataset_info->dataset->path[2]
        path = dataset_info['path'][2]  # patch
        self.patch_dir = os.path.join(dataset_path, path)

        self.is_shuffle = train_conf['shuffle']

        # define shuffle list outside
        if i_dataset is None or i_dataset == 'all':
            search_path = self.patch_dir + '/*.npz'
        else:
            search_path = self.patch_dir + '/{}-*.npz'.format(i_dataset)
        self.patch_lib_filenames = np.array(sorted(glob.glob(search_path)))
        self.patch_lib_filenames = self.patch_lib_filenames[shuffle_order]

        # expanded patch lib = N
        self.patch_lib_filenames = np.tile(self.patch_lib_filenames, math.ceil(self.N / len(self.patch_lib_filenames)))[:self.N]

        # random shuffle
        if self.is_shuffle is True:
            np.random.shuffle(self.patch_lib_filenames)

        # read mean/std without norm 13/10
        self.mean, self.std = read_dataloader_meanstd(gen_conf, train_conf, i_dataset)


    def __len__(self):
        return math.ceil(self.N / self.batch_size)

    def __getitem__(self, idx):
        batch_filenames = self.patch_lib_filenames[idx * self.batch_size:(idx+1)*self.batch_size]

        x_batch = []
        y_batch = []
        for filename in batch_filenames:
            x_patches, y_patches = self.__read_patch_data(filename)
            rnd_patch_idx = np.random.randint(len(x_patches), size=1)
            x_batch.append(x_patches[np.asscalar(rnd_patch_idx)])
            y_batch.append(y_patches[np.asscalar(rnd_patch_idx)])

        # normalize batch
        x_batch = self.__normalise_set(np.array(x_batch), self.modalities, self.mean['input'], self.std['input'])
        y_batch = self.__normalise_set(np.array(y_batch), self.modalities, self.mean['output'], self.std['output'])

        return x_batch, y_batch

    def on_epoch_end(self):
        # random shuffle
        if self.is_shuffle is True:
            np.random.shuffle(self.patch_lib_filenames)

    def __normalise_set(self, input_data, num_modalities, mean, std):
        input_data_tmp = np.copy(input_data)
        for vol_idx in range(len(input_data_tmp)):
            for modality in range(num_modalities):
                input_data_tmp[vol_idx, modality] -= mean[modality]
                input_data_tmp[vol_idx, modality] /= std[modality]
        return input_data_tmp

    def __unzip(self, src_filename, dest_dir):
        with zipfile.ZipFile(src_filename) as z:
            z.extractall(path=dest_dir)

    def clear_extracted_files(self):
        if os.path.isdir(self.patch_dir):
            shutil.rmtree(self.patch_dir)
            return True
        else:
            print("'{}' doesn't exist ...".format(self.patch_dir))
            return False

    def __read_patch_data(self, filepath):
        files = np.load(filepath)
        return files['x_patches'], files['y_patches']

    def __split_train_val(self, train_indexes, N, validation_split, is_val_gen = False):
        N_vol = len(train_indexes)
        val_volumes = np.int32(np.ceil(N_vol * validation_split))
        train_volumes = N_vol - val_volumes

        val_patches = np.int32(np.ceil(N * validation_split))
        train_patches = N-val_patches

        if is_val_gen is True and validation_split != 0:
            return train_indexes[train_volumes:], val_patches
        elif is_val_gen is False:
            return train_indexes[:train_volumes], train_patches # training data
        else:
            raise ValueError("validation_split should be non-zeroed value when is_val_gen == True")