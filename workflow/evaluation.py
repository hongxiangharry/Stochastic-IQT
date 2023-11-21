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
""" Workflow for evaluation. """

from utils.image_evaluation import image_evaluation, interp_evaluation, image_evaluation_lowcost, image_evaluation_lowcost_uncertainty
import itertools

def eval_contrast_uncertainty(gen_conf, test_conf, sim_norm_conf, originProcessFlag='origin', isClearSingleOutputs=False, is_agg = True):
    if 'contrast_type' not in sim_norm_conf['sim'].keys():
        print('Use contrast type as mc ..')
        sim_norm_conf['sim']['contrast_type'] = 'mc'

    if sim_norm_conf['sim']['contrast_type'] == 'rc' or sim_norm_conf['sim']['contrast_type'] == 'random-contrast':
        print('Run eval on RC ...')
        eval_random_contrast_uncertainty(gen_conf, test_conf, originProcessFlag=originProcessFlag, isClearSingleOutputs=isClearSingleOutputs, is_agg = is_agg)
    else:
        raise Exception("contrast type has not been defined ...")

    return True

def eval_random_contrast_uncertainty(gen_conf, test_conf, originProcessFlag='origin', isClearSingleOutputs=False, is_agg = True):
    num_sim_train = test_conf['num_dataset']
    num_sim_test = 1 # test_conf['num_test_dataset']
    iters = range(num_sim_train)

    if isClearSingleOutputs is False:
        # eval on individual model
        for i_dataset in iters:
            image_evaluation_lowcost_uncertainty(gen_conf, test_conf, i_dataset, originProcessFlag=originProcessFlag)

    if test_conf['ensemble_type'] is not None and is_agg is True:
        # eval on ensemble model
        image_evaluation_lowcost_uncertainty(gen_conf, test_conf, test_conf['ensemble_type'], originProcessFlag=originProcessFlag)
    return True

def eval_contrast(gen_conf, test_conf, sim_norm_conf, originProcessFlag='origin', isClearSingleOutputs=False):
    if 'contrast_type' not in sim_norm_conf['sim'].keys():
        print('Use contrast type as mc ..')
        sim_norm_conf['sim']['contrast_type'] = 'mc'

    if sim_norm_conf['sim']['contrast_type'] == 'mc' or sim_norm_conf['sim']['contrast_type'] == 'multi-contrast':
        print('Run eval on MC ...')
        eval_multi_contrast(gen_conf, test_conf, originProcessFlag=originProcessFlag, isClearSingleOutputs=isClearSingleOutputs)
    elif sim_norm_conf['sim']['contrast_type'] == 'rc' or sim_norm_conf['sim']['contrast_type'] == 'random-contrast':
        print('Run eval on RC ...')
        eval_random_contrast(gen_conf, test_conf, originProcessFlag=originProcessFlag, isClearSingleOutputs=isClearSingleOutputs)
    else:
        raise Exception("contrast type has not been defined ...")

    return True

def eval_random_contrast(gen_conf, test_conf, originProcessFlag='origin', isClearSingleOutputs=False):
    num_sim_train = test_conf['num_dataset']
    # num_sim_test = 1 # test_conf['num_test_dataset']
    iters = range(num_sim_train)

    if isClearSingleOutputs is False:
        # eval on individual model
        for i_dataset in iters:
            image_evaluation_lowcost(gen_conf, test_conf, i_dataset, originProcessFlag=originProcessFlag)

    if test_conf['ensemble_type'] is not None:
        # eval on ensemble model
        image_evaluation_lowcost(gen_conf, test_conf, test_conf['ensemble_type'], originProcessFlag=originProcessFlag)
    return True


def eval_multi_contrast(gen_conf, test_conf, originProcessFlag='origin', isClearSingleOutputs=False):
    num_sim_train = test_conf['num_dataset']
    num_sim_test = test_conf['num_test_dataset']
    iters = itertools.product(range(num_sim_train), range(num_sim_test))

    if isClearSingleOutputs is False:
        # eval on individual model
        for i_dataset in iters:
            image_evaluation_lowcost(gen_conf, test_conf, i_dataset, originProcessFlag=originProcessFlag)

    if test_conf['ensemble_type'] is not None:
        # eval on ensemble model
        image_evaluation_lowcost(gen_conf, test_conf, test_conf['ensemble_type'], originProcessFlag=originProcessFlag)
    return True


def evaluation(gen_conf, test_conf, case_name = None):
    image_evaluation(gen_conf, test_conf, case_name)

def interp_eval(gen_conf, test_conf, case_name = 1):
    interp_evaluation(gen_conf, test_conf, case_name)
