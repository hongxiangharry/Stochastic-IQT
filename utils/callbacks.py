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
""" Call back functions during training."""

from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from utils.ioutils import generate_output_filename

import numpy as np
import os

def generate_callbacks(general_configuration, training_configuration, case_name) :
            
    ## save model
    model_filename = generate_output_filename(
        general_configuration['model_path'],
        training_configuration['dataset'],
        case_name,
        training_configuration['approach'],
        training_configuration['dimension'],
        str(training_configuration['patch_shape']),
        str(training_configuration['extraction_step']),
        'h5')
    
    if (os.path.exists(model_filename) == False) or (training_configuration['retrain'] == True):
        callbacks_arr = []

        ## check and make folders
        model_foldername = os.path.dirname(model_filename)
        if not os.path.isdir(model_foldername) :
            os.makedirs(model_foldername)

        csv_filename = generate_output_filename(
            general_configuration['log_path'],
            training_configuration['dataset'],
            case_name,
            training_configuration['approach'],
            training_configuration['dimension'],
            str(training_configuration['patch_shape']),
            str(training_configuration['extraction_step']),
            'cvs')
        ## check and make folders
        csv_foldername = os.path.dirname(csv_filename)
        if not os.path.isdir(csv_foldername) :
            os.makedirs(csv_foldername)

        if training_configuration['validation_split'] != 0 :
            stopper = EarlyStopping(
                patience=training_configuration['patience'])
            callbacks_arr.append(stopper)

            checkpointer = ModelCheckpoint(
                filepath=model_filename,
                verbose=0,
                save_best_only=True,
                save_weights_only=True)
            callbacks_arr.append(checkpointer)

        csv_logger = CSVLogger(csv_filename, separator=',')
        callbacks_arr.append(csv_logger)

        if training_configuration['optimizer'] == 'SGD' :
            def step_decay(epoch) :
                initial_lr = training_configuration['learning_rate']
                drop = training_configuration['decay']
                epochs_drop = 5.0
                lr = initial_lr * (drop ** np.floor((1 + epoch) / epochs_drop))
                return lr
            lr_scheduler = LearningRateScheduler(step_decay, verbose=1)
            callbacks_arr.append(lr_scheduler)
        return callbacks_arr
    else :
        return None
