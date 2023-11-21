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
""" Workflow for simulation and normalization. """

from utils.prob_sim_utils import generate_sim
from utils.norm_utils import merged_normalisation_random_contrast as merged_normalisation

def sim_norm(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag='train', normType=None, out_suffix = '_norm', train_conf = None):
    if 'normType' in sim_norm_conf['norm'].keys() and normType is None:
        normType = sim_norm_conf['norm']['normType']
    gen_conf, traintest_conf, sim_norm_conf = run_sim_norm(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag, normType, out_suffix = out_suffix, train_conf = train_conf)

    return gen_conf, traintest_conf, sim_norm_conf

def run_sim_norm(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag = 'train', normType='merge',
                 out_suffix = '_norm', train_conf=None):
    dataset = traintest_conf['dataset']
    if dataset in ['HCP', 'HCP-Wu-Minn-Contrast-Multimodal', 'HCP-Wu-Minn-Contrast-Multimodal-test',
                   'HCP-rc', 'HCP-rc-test', 'MBB-rc', 'MBB-rc-test']:
        print("Processing simulation step:")
        gen_conf, traintest_conf, sim_norm_conf = generate_sim(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag)
        sim_conf = sim_norm_conf['sim']

        if sim_conf['is_sim'] is False:
            if normType == 'merge':
                print("Processing normalisation step:")
                traintest_conf['num_test_dataset'] = 1  # new 2/6
                merged_normalisation(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag, train_conf=train_conf, out_suffix = out_suffix)
    elif dataset in ['Nigeria19-rc', 'Nigeria17-rc']:
        if trainTestFlag == 'test':
            sim_conf = sim_norm_conf['sim']

            if sim_conf['is_sim'] is False:
                if normType == 'merge':
                    print("Processing normalisation step:")
                    traintest_conf['num_test_dataset'] = 1  # new 2/6
                    merged_normalisation(gen_conf, traintest_conf, sim_norm_conf, trainTestFlag, train_conf=train_conf,
                                         out_suffix = out_suffix, is_background_masking=True)
        else:
            raise ValueError("trainTestFlag should be test for the Nigeria'19 dataset.")


    return gen_conf, traintest_conf, sim_norm_conf