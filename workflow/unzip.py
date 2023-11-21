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
""" Workflow for unzip. """

import zipfile
import os
import shutil

def unzip_patchlib(gen_conf, train_conf):
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    # dest dir for unzipping patch.zip
    dest_patch_dir = os.path.join(dataset_path, dataset_info['path'][2])

    # src path of patch.zip
    patch_zip_dir = os.path.join(dataset_path, dataset_info['path'][3] )
    os.chdir(patch_zip_dir)  # change working directory to 'patch_zip_dir'

    patch_filename_pattern = train_conf['patch_filename_pattern']
    extraction_step = train_conf['extraction_step']
    patch_shape = train_conf['patch_shape']  # input patch shape
    num_volumes = train_conf['num_train_sbj']  # actual num of train sbj used to, 15
    patch_zip_filename = patch_filename_pattern.format('all', patch_shape, extraction_step, num_volumes, 'zip')

    __unzip(patch_zip_filename, dest_patch_dir)

    return True

def remove_unzip_folder(gen_conf, train_conf):
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    # dest dir for unzipping patch.zip
    dest_patch_dir = os.path.join(dataset_path, dataset_info['path'][2])
    shutil.rmtree(dest_patch_dir, ignore_errors=True)
    return True

def __unzip(src_filename, dest_dir):
    with zipfile.ZipFile(src_filename) as z:
        z.extractall(path=dest_dir)