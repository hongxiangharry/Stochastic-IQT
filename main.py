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
""" Main function """
import argparse
from config.config import general_configuration as gen_conf
from config.config import training_configuration as train_conf
from config.config import test_configuration as test_conf
from config.config import simulation_configuration_train as sim_train_conf
from config.config import simulation_configuration_test as sim_test_conf
from workflow.data_preparation import data_preparation
from workflow.sim_norm import sim_norm
from workflow.unzip import unzip_patchlib, remove_unzip_folder

from workflow.build_training_set import build_training_set_contrasts
from workflow.train import train_single_net_multi_contrast
from workflow.test import test_contrast
from workflow.evaluation import eval_contrast


def main(gen_conf, train_conf, test_conf, sim_train_conf, sim_test_conf, is_test=False):
    if is_test is False:
        # training process
        opt, gen_conf, train_conf, sim_train_conf = data_preparation(gen_conf, train_conf, sim_train_conf, trainTestFlag='train')

        gen_conf, train_conf, sim_train_conf = sim_norm(gen_conf, train_conf, sim_norm_conf=sim_train_conf, trainTestFlag='train',
                                                        out_suffix = '')

        count_patches_arr = build_training_set_contrasts(gen_conf, train_conf, sim_train_conf) # build training set
        print('Number of patch for training: ', count_patches_arr)

        unzip_patchlib(gen_conf, train_conf)

        train_single_net_multi_contrast(gen_conf, train_conf)

    # test process
    opt, gen_conf, test_conf, sim_test_conf = data_preparation(gen_conf, test_conf, sim_test_conf, trainTestFlag='eval')

    gen_conf, test_conf, sim_test_conf = sim_norm(gen_conf, test_conf, sim_norm_conf=sim_test_conf, trainTestFlag='eval',
                                                 out_suffix = '', train_conf=train_conf)

    isClearSingleOutputs = False

    test_contrast(gen_conf, test_conf, sim_test_conf, train_conf=train_conf, trainTestFlag='eval',
                  isClearSingleOutputs=isClearSingleOutputs)

    # evaluation
    eval_contrast(gen_conf, test_conf, sim_test_conf, originProcessFlag='process', isClearSingleOutputs=isClearSingleOutputs)

    if is_test is False:
        remove_unzip_folder(gen_conf, train_conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SIQT-Tensorflow-Version.')
    parser.add_argument('--test', action='store_true', help='Run only test mode.')
    arg = parser.parse_args()

    main(gen_conf, train_conf, test_conf, sim_train_conf, sim_test_conf, is_test = arg['test'])