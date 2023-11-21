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
""" Workflow for building training patch set. """

from utils.build_patch_lib_utils import build_patch_lib, build_patch_lib_random_contrasts

## evaluate_using_training_testing_split
def build_training_set(gen_conf, train_conf) :
    train_conf['actual_num_patches'] = []
    for i_dataset in range(train_conf['num_dataset']):
        count = build_patch_lib(gen_conf, train_conf, i_dataset)
        train_conf['actual_num_patches'].append( count )
    return train_conf['actual_num_patches']

def build_training_set_contrasts(gen_conf, train_conf, sim_norm_conf = None):
    if sim_norm_conf is None:
        sim_norm_conf['sim']['contrast_type'] = 'mc'
    elif 'contrast_type' not in sim_norm_conf['sim'].keys():
        sim_norm_conf['sim']['contrast_type'] = 'mc'

    if sim_norm_conf['sim']['contrast_type'] == 'mc':
        count = build_training_set_multi_contrasts(gen_conf, train_conf)
    elif sim_norm_conf['sim']['contrast_type'] == 'rc':
        count = build_training_set_random_contrasts(gen_conf, train_conf)

    return count


def build_training_set_multi_contrasts(gen_conf, train_conf):
    count = build_patch_lib(gen_conf, train_conf, 'all')
    train_conf['actual_num_patches'] = count
    return count

def build_training_set_random_contrasts(gen_conf, train_conf):
    count = build_patch_lib_random_contrasts(gen_conf, train_conf, 'all')
    train_conf['actual_num_patches'] = count
    return count