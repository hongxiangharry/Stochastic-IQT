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
""" Workflow for training stage. """

import numpy as np
from architectures.arch_creator import generate_model
from utils.callbacks import generate_callbacks
from utils.ioutils import read_model, save_msecorr_array
from utils.visualize_model import visualize_model
import matplotlib.pyplot as plt
from utils.preprocessing_util import preproc_input
from utils.ioutils import DefineTrainValDataLoader
from utils.ioutils import generate_output_filename

def train_single_net_random_contrast(gen_conf, train_conf):
    mse_array = []

    print("Set up data loader for all simulated images ...")
    train_generator, val_generator = DefineTrainValDataLoader(gen_conf, train_conf, i_dataset='all')

    x, y = train_generator[0]
    print(x.shape, y.shape)

    # for case in range(train_conf['cases']):
    print("Start training on all simulated datasets ...")
    model, mse = train_model_generator(gen_conf, train_conf, train_generator, val_generator, case_name='all')
    mse_array.append(mse)

    # write to file
    save_msecorr_array(gen_conf, train_conf, mse_array, None)

    return mse_array

def train_single_net_multi_contrast(gen_conf, train_conf):
    mse_array = []

    print("Set up data loader for all simulated images ...")
    train_generator, val_generator = DefineTrainValDataLoader(gen_conf, train_conf, i_dataset='all')

    x, y = train_generator[0]
    print(x.shape, y.shape)

    # for case in range(train_conf['cases']):
    print("Start training on all simulated datasets ...")
    model, mse = train_model_generator(gen_conf, train_conf, train_generator, val_generator, case_name='all')
    mse_array.append(mse)

    # write to file
    save_msecorr_array(gen_conf, train_conf, mse_array, None)

    return mse_array

def train_multi_contrast(gen_conf, train_conf) :
    mse_array_set = []
    for i_dataset in range(train_conf['num_dataset']):
        mse_array = training(gen_conf, train_conf, i_dataset)
        mse_array_set.append(mse_array)

    return mse_array_set

## evaluate_using_training_testing_split
def training(gen_conf, train_conf, i_dataset = None) :
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]

    mse_array = []

    ## interpolation
    if dataset_info['is_preproc'] == True:
        print("Interpolating training data...")
        interp_order = dataset_info['interp_order']
        preproc_input(gen_conf, train_conf, is_training=True, interp_order=interp_order) # input pre-processing

    print("Set up data loader ...")
    train_generator, val_generator = DefineTrainValDataLoader(gen_conf, train_conf, i_dataset=i_dataset)

    x, y = train_generator[0]
    print(x.shape, y.shape)

    # for case in range(train_conf['cases']):
    print("Start training on the " + str(i_dataset) + " th dataset(s) ...")
    model, mse = train_model_generator(gen_conf, train_conf, train_generator, val_generator, case_name=i_dataset)
    mse_array.append(mse)

    # write to file
    save_msecorr_array(gen_conf, train_conf, mse_array, None)
    # train_generator.clear_extracted_files() # remove the extracted patch folder
    # update[25/08], the extracted files will be cleared outside the iqt code

    return mse_array

def train_model_generator(gen_conf, train_conf, train_generator, val_generator, case_name):
    if train_conf['num_epochs'] != 0:
        # training callbacks
        callbacks = generate_callbacks(gen_conf, train_conf, case_name)
        if callbacks is not None: # train or not ?
            model = __train_model(gen_conf, train_conf, train_generator, val_generator, case_name, callbacks)
        else:
            model = read_model(gen_conf, train_conf, case_name)

        ## compute mse and corr on validation set
        mse = model.evaluate(val_generator,
                             batch_size=train_conf['batch_size'],
                             workers=train_conf['workers'],
                             use_multiprocessing=train_conf['use_multiprocessing'],
                             verbose=train_conf['verbose'])
    else:
        model = []
        mse = None

    return model, mse

def __train_model(gen_conf, train_conf, train_generator, val_generator, case_name, callbacks, vis_flag=False):
    model = generate_model(gen_conf, train_conf)

    print(model.summary()) # print model summary

    history = model.fit(
        train_generator,
        batch_size=train_conf['batch_size'],
        epochs=train_conf['num_epochs'],
        validation_data=val_generator,
        shuffle=train_conf['shuffle'],
        verbose=train_conf['verbose'],
        workers=train_conf['workers'],
        use_multiprocessing=train_conf['use_multiprocessing'],
        callbacks=callbacks)
    if train_conf['validation_split'] == 0:
        ## save model
        model_filename = generate_output_filename(
            gen_conf['model_path'],
            train_conf['dataset'],
            case_name,
            train_conf['approach'],
            train_conf['dimension'],
            str(train_conf['patch_shape']),
            str(train_conf['extraction_step']),
            'h5')
        model.save_weights(model_filename, save_format='h5')
    print(history.params) # print model parameters

    # if vis_flag == True:
    visualize_model(model, history, gen_conf, train_conf, case_name, vis_flag)

    return model

def split_train_val(train_indexes, validation_split) :
    N = len(train_indexes)
    val_volumes = np.int32(np.ceil(N * validation_split))
    train_volumes = N - val_volumes

    return train_indexes[:train_volumes], train_indexes[train_volumes:]

def compute_statistics(input_data, num_modalities) :
    mean = np.zeros((num_modalities, ))
    std = np.zeros((num_modalities, ))

    for modality in range(num_modalities) :
        modality_data = input_data[:, modality]
        mean[modality] = np.mean(modality_data)
        std[modality] = np.std(modality_data)

    return mean, std

def normalise_set(input_data, num_modalities, mean, std) :
    input_data_tmp = np.copy(input_data)
    for vol_idx in range(len(input_data_tmp)) :
        for modality in range(num_modalities) :
            input_data_tmp[vol_idx, modality] -= mean[modality]
            input_data_tmp[vol_idx, modality] /= std[modality]
    return input_data_tmp

def debug_plot_patch(input_patch, output_patch):

    fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(9.3, 4),
                            subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)

    for idx, ax in enumerate(axs.flat[:6]):
        ax.imshow(input_patch[idx+1000, 0, 16, :, :], interpolation=None, cmap='gray', aspect=0.25)
        # ax.set_title(str(interp_method))
    for idx, ax in enumerate(axs.flat[6:]):
        ax.imshow(output_patch[idx+1000, 0, 16, :, :], interpolation=None, cmap='gray')

    plt.tight_layout()
    plt.show()
