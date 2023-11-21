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
""" Set up config data structure. """

import os
import argparse
import csv
import glob

def argument_parse():
    # -------------------------- Set up configurations ----------------------------
    # Basic settings
    ## description: text to display before the argument help
    parser = argparse.ArgumentParser(description='SIQT-Tensorflow-Version.')
    ## dest : The name of the attribute to be added to the object returned by parse_args()
    ## If there is no explicit written "dest" parameter, the key should be "e" in this case.

    # default: None for '?' and [] for '*'
    # list to tuple
    # system conf
    parser.add_argument('--gpu', type=str, default="0", help='which GPU to use') # not used
    parser.add_argument('--data_format', dest='data_format', type=str, default="channels_first",
                        help='determine the order of channel in the CNN: channels_first/channels_last')

    ## directory
    parser.add_argument('-dp', '--dataset_path', dest='dataset_path', nargs='?', type=str, help='dataset directory')
    parser.add_argument('-bp', '--base_path', dest='base_path', nargs='?', type=str, help='workplace directory')
    parser.add_argument('-jn', '--job_name', dest='job_name', nargs='?', type=str, help='job name of folder')
    parser.add_argument('-j', '--job_id', dest='job_id', nargs='?', default=None, type=str, help='job id to qsub system')

    ## dataset info
    parser.add_argument('--dataset', dest='dataset', nargs='?', type=str, help='test dataset name')
    parser.add_argument('--no_subject', dest='no_subject', nargs='*', type=int, help='set train/test subjects')
    parser.add_argument('--num_samples', dest='num_samples', nargs='*', type=int, help='set augmenting total/train/test samples')
    # patching info
    parser.add_argument('-es', '--extraction_step', dest='extraction_step', nargs='*', type=int,
                        help='stride between patch for training')
    parser.add_argument('-est', '--extraction_step_test', dest='extraction_step_test', nargs='*', type=int,
                        help='stride between patch for testing')
    parser.add_argument('-ip', '--input_patch', dest='input_patch', nargs='*', type=int,
                        help='input patch shape')
    parser.add_argument('-op', '--output_patch', dest='output_patch', nargs='*', type=int,
                        help='output patch shape')
    parser.add_argument('-opt', '--output_patch_test', dest='output_patch_test', nargs='*', type=int,
                        help='output patch shape for testing')
    parser.add_argument('-psr', '--patch_sampling_rate', dest='patch_sampling_rate', nargs='?', type=float,
                        help='rate of patch sampling per image')
    parser.add_argument('-mnp', '--max_num_patches', dest='max_num_patches', nargs='?', type=int,
                        help='maximal number of patches per volume')
    parser.add_argument('-mnppf', '--max_num_patches_per_file', dest='max_num_patches_per_file', nargs='?', type=int,
                        help='maximal number of patches per file')

    parser.add_argument('-ntp', '--num_training_patches', dest='num_training_patches', nargs='?', type=int,
                        help='number of training patches (deprecated)') # 23-06-06 not used
    parser.add_argument('--rebuild', action='store_true', help='rebuild training patch set')

    parser.add_argument('-pfp', '--patch_filename_pattern', dest='patch_filename_pattern', nargs='?', type=str, help='patch filename pattern')
    parser.add_argument('-ppn', '--preprocess_num', dest='preprocess_num', nargs='?', type=str, help='preprocessing method number')

    # network info
    parser.add_argument('--approach', dest='approach', nargs='?', type=str, help='name of network architecture')
    parser.add_argument('-l', '--loss', dest='loss', nargs='?', type=str, help='name of loss')
    parser.add_argument('-lp', '--loss_params', dest='loss_params', nargs='*', type=float, help='parameters for loss function')
    parser.add_argument('-ne', '--no_epochs', dest='no_epochs', nargs='?', type=int, help='number of epochs')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', nargs='?', type=int, help='batch size')
    parser.add_argument('--patience', dest='patience', nargs='?', type=int, help='early stop at patience number')
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', nargs='?', type=float, help='learning rate')
    parser.add_argument('-dc', '--decay', dest='decay', nargs='?', type=float, help='decay of learning rate')
    parser.add_argument('-dsf', '--downsize_factor', dest='downsize_factor', nargs='?', type=int, help='downsize factor for CNN')
    parser.add_argument('-nk', '--num_kernels', dest='num_kernels', nargs='?', type=int, help='number of kernels per block')
    parser.add_argument('-nf', '--num_filters', dest='num_filters', nargs='?', type=int,
                        help='number of filters per conv layer')
    parser.add_argument('-mt', '--mapping_times', dest='mapping_times', nargs='?', type=int,
                        help='number of FSRCNN shrinking layers')
    parser.add_argument('-nl', '--num_levels', dest='num_levels', nargs='?', type=int,
                        help='number of U-Net levels')
    parser.add_argument('-c', '--cases', dest='cases', nargs='?', type=int, help='number of training cases')
    parser.add_argument('-nts', '--num_train_sbj', dest='num_train_sbj', nargs='?', type=int,
                        help='number of training subjects')
    parser.add_argument('-vsr', '--validation_split_ratio', dest='validation_split_ratio', nargs='?', type=float,
                        help='train-validation split ratio')

    parser.add_argument('-nen', '--num_ensembles', dest='num_ensembles', nargs='?', type=int,
                        help='number of ensemble models using masksembles')
    parser.add_argument('-mss', '--ms_scale', dest='ms_scale', nargs='?', type=float,
                        help='the scale for masksembles')

    # action : Turn on the value for the key, i.e. "overwrite=True"
    parser.add_argument('--retrain', action='store_true', help='restart the training completely')

    arg = parser.parse_args()
    return vars(arg)  ## return a dictionary type of arguments and the values.

def set_conf_info(gen_conf, train_conf, trainTestFlag='train'):
    opt = argument_parse() # read parser from the command line

    assert opt['data_format'] in ['channels_first', 'channels_last'], 'the argument data_format only allows channels_first/channels_last'
    train_conf['data_format'] = opt['data_format']

    if opt['dataset_path'] is not None: gen_conf['dataset_path'] = opt['dataset_path']
    if opt['base_path'] is not None: gen_conf['base_path'] = opt['base_path']
    if opt['job_name'] is not None: gen_conf['job_name'] = opt['job_name']
    if opt['job_id'] is not None: gen_conf['job_id'] = opt['job_id']

    if opt['dataset'] is not None: # to preserve train_conf['dataset'] in config and test_conf['dataset'] by declare
        if trainTestFlag != 'train': # declare name of test set when train and test sets are different.
            train_conf['dataset'] = opt['dataset']
    if opt['no_subject'] is not None: gen_conf['dataset_info'][train_conf['dataset']]['num_volumes'] = opt['no_subject']
    if opt['num_samples'] is not None: gen_conf['dataset_info'][train_conf['dataset']]['num_samples'] = opt['num_samples']

    if opt['extraction_step'] is not None: train_conf['extraction_step'] = tuple(opt['extraction_step'])
    if opt['extraction_step_test'] is not None: train_conf['extraction_step_test'] = tuple(opt['extraction_step_test'])
    if opt['input_patch'] is not None: train_conf['patch_shape'] = tuple(opt['input_patch'])
    if opt['output_patch'] is not None: train_conf['output_shape'] = tuple(opt['output_patch'])
    if opt['output_patch_test'] is not None: train_conf['output_shape_test'] = tuple(opt['output_patch_test'])
    if opt['patch_sampling_rate'] is not None: train_conf['patch_sampling_rate'] = opt['patch_sampling_rate']
    if opt['max_num_patches'] is not None: train_conf['max_num_patches'] = opt['max_num_patches']
    if opt['max_num_patches_per_file'] is not None: train_conf['max_num_patches_per_file'] = opt['max_num_patches_per_file']
    if opt['patch_filename_pattern'] is not None: train_conf['patch_filename_pattern'] = opt[
        'patch_filename_pattern']
    if opt['preprocess_num'] is not None: gen_conf['preprocess_num'] = opt[
        'preprocess_num']
    if opt['num_training_patches'] is not None: train_conf['num_training_patches'] = opt['num_training_patches']
    if opt['num_train_sbj'] is not None: train_conf['num_train_sbj'] = opt['num_train_sbj']
    if opt['validation_split_ratio'] is not None: train_conf['validation_split'] = opt['validation_split_ratio']

    train_conf['rebuild'] = opt['rebuild']

    if opt['approach'] is not None: train_conf['approach'] = opt['approach']
    if opt['loss'] is not None: train_conf['loss'] = opt['loss']
    if opt['loss_params'] is not None: train_conf['loss_params'] = opt['loss_params']
    if opt['no_epochs'] is not None: train_conf['num_epochs'] = opt['no_epochs']
    if opt['batch_size'] is not None: train_conf['batch_size'] = opt['batch_size']
    if opt['patience'] is not None: train_conf['patience'] = opt['patience']

    if opt['learning_rate'] is not None: train_conf['learning_rate'] = opt['learning_rate']
    if opt['decay'] is not None: train_conf['decay'] = opt['decay']
    if opt['downsize_factor'] is not None: train_conf['downsize_factor'] = opt['downsize_factor']
    if opt['num_kernels'] is not None: train_conf['num_kernels'] = opt['num_kernels']
    if opt['num_filters'] is not None: train_conf['num_filters'] = opt['num_filters']
    if opt['num_ensembles'] is not None: train_conf['num_ensembles'] = opt['num_ensembles']
    if opt['mapping_times'] is not None: train_conf['mapping_times'] = opt['mapping_times']
    if opt['num_levels'] is not None: train_conf['num_levels'] = opt['num_levels']
    if opt['cases'] is not None: train_conf['cases'] = opt['cases']

    if opt['ms_scale'] is not None: train_conf['ms_scale'] = opt['ms_scale']

    train_conf['retrain'] = opt['retrain']
    return opt, gen_conf, train_conf

def conf_dataset(gen_conf, train_conf, simnorm_conf=None, trainTestFlag='train'):
    if gen_conf['base_path'] is None:
        gen_conf['base_path'] = os.getcwd()
    if gen_conf['dataset_path'] is None:
        gen_conf['dataset_path'] = os.path.join(gen_conf['base_path'], 'data')

    # configure log/model/result/evaluation paths.
    gen_conf['log_path'] = os.path.join(gen_conf['base_path'], gen_conf['job_name'], gen_conf['log_path_'])
    gen_conf['model_path'] = os.path.join(gen_conf['base_path'], gen_conf['job_name'], gen_conf['model_path_'])
    gen_conf['results_path'] = os.path.join(gen_conf['base_path'], gen_conf['job_name'], gen_conf['results_path_'])
    gen_conf['evaluation_path'] = os.path.join(gen_conf['base_path'], gen_conf['job_name'], gen_conf['evaluation_path_'])

    dataset = train_conf['dataset']

    if dataset in ['HCP', 'HCP-Wu-Minn-Contrast-Multimodal', 'HCP-Wu-Minn-Contrast-Multimodal-test', 'HCP-LRTV', 'HCP-cubic', 'HCP-GT', 'HCP-input'] :
        return conf_Multimodal_dataset(gen_conf, train_conf, simnorm_conf, trainTestFlag)
    if dataset in ['Nigeria19-rc', 'Nigeria17-rc'] :
        return conf_n19_rc_dataset(gen_conf, train_conf, simnorm_conf, trainTestFlag)
    if dataset in ['HCP-rc', 'HCP-rc-test'] :
        return conf_HCP_rc_dataset(gen_conf, train_conf, simnorm_conf, trainTestFlag)
    if dataset in ['MBB-rc', 'MBB-rc-test'] :
        return conf_MBB_rc_dataset(gen_conf, train_conf, simnorm_conf, trainTestFlag)

def conf_n19_rc_dataset(gen_conf, train_conf, simnorm_conf=None, trainTestFlag='train'):
    dataset_path = gen_conf['dataset_path']
    if 'preprocess_num' in gen_conf.keys():
        dataset_name = train_conf['dataset']+'-'+gen_conf['preprocess_num']
        if dataset_name not in gen_conf['dataset_info'].keys():
            dataset_name = train_conf['dataset']
        if 'patch_filename_pattern' in train_conf.keys():
            train_conf['patch_filename_pattern'] = gen_conf['preprocess_num'] + '-' + train_conf['patch_filename_pattern']
        if simnorm_conf is not None:
            sim_name = 'sim-' + gen_conf['preprocess_num']
            if sim_name in simnorm_conf.keys():
                simnorm_conf['sim'] = simnorm_conf['sim-'+gen_conf['preprocess_num']]
            if 'seed' in simnorm_conf['sim'].keys():
                simnorm_conf['sim']['seed'] = simnorm_conf['sim']['seed'] * int(gen_conf['preprocess_num']) + 1 # fix seed
    else:
        dataset_name = train_conf['dataset']

    dataset_info = gen_conf['dataset_info'][dataset_name]

    if 'preprocess_num' in gen_conf.keys():
        dataset_info['path'][0] = dataset_info['path'][0].format(gen_conf['preprocess_num'])

    path = dataset_info['path'][1] # original dir
    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]

    whole_dataset_path = os.path.join(dataset_path, path)
    subject_lib = sorted([os.path.basename(subject) for subject in glob.glob(whole_dataset_path+'/*')])

    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    dataset_info['training_subjects'] = []
    idx_sn = 0
    for subject in subject_lib:
        if idx_sn < dataset_info['num_volumes'][0]:
            dataset_info['training_subjects'].append(subject)
            idx_sn += 1

    dataset_info['test_subjects'] = []
    for subject in subject_lib[idx_sn:]:
        if idx_sn < dataset_info['num_volumes'][0] + dataset_info['num_volumes'][1]:
            dataset_info['test_subjects'].append(subject)
            idx_sn += 1

    # set process data train/test path for dataset_info['path'][0] ..
    if trainTestFlag == 'train':
        new_traintest_folder = 'train'
        counter_traintest_folder = 'test'
    else:
        new_traintest_folder = 'test'
        counter_traintest_folder = 'train'
    curr_traintest_folder = os.path.basename(dataset_info['path'][0])

    if curr_traintest_folder != 'train' and curr_traintest_folder != 'test':
        dataset_info['path'][0] = os.path.join(dataset_info['path'][0], new_traintest_folder)
    elif curr_traintest_folder == counter_traintest_folder:
        dataset_info['path'][0] = os.path.join(os.path.dirname(dataset_info['path'][0]), new_traintest_folder)

    gen_conf['dataset_info'][train_conf['dataset']] = dataset_info

    return gen_conf, train_conf, simnorm_conf

def conf_MBB_rc_dataset(gen_conf, train_conf, simnorm_conf=None, trainTestFlag='train'):
    dataset_path = gen_conf['dataset_path']
    if 'preprocess_num' in gen_conf.keys():
        dataset_name = train_conf['dataset']+'-'+gen_conf['preprocess_num']
        if dataset_name not in gen_conf['dataset_info'].keys():
            dataset_name = train_conf['dataset']
        train_conf['patch_filename_pattern'] = gen_conf['preprocess_num'] + '-' + train_conf['patch_filename_pattern']
        if simnorm_conf is not None:
            sim_name = 'sim-' + gen_conf['preprocess_num']
            if sim_name in simnorm_conf.keys():
                simnorm_conf['sim'] = simnorm_conf['sim-'+gen_conf['preprocess_num']]
            if 'seed' in simnorm_conf['sim'].keys():
                simnorm_conf['sim']['seed'] = simnorm_conf['sim']['seed'] * int(gen_conf['preprocess_num']) + 1 # fix seed
    else:
        dataset_name = train_conf['dataset']
    job_id = gen_conf['job_id'] if 'job_id' in gen_conf.keys() else None

    dataset_info = gen_conf['dataset_info'][dataset_name]

    if 'preprocess_num' in gen_conf.keys():
        dataset_info['path'][0] = dataset_info['path'][0].format(gen_conf['preprocess_num'])

    path = dataset_info['path'][1] # original dir
    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]

    whole_dataset_path = os.path.join(dataset_path, path)
    subject_lib = sorted([os.path.basename(subject) for subject in glob.glob(whole_dataset_path+'/*')])

    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    dataset_info['training_subjects'] = []
    idx_sn = 0
    for subject in subject_lib:
        if idx_sn < dataset_info['num_volumes'][0]:
            dataset_info['training_subjects'].append(subject)
            idx_sn += 1

    dataset_info['test_subjects'] = []
    for subject in subject_lib[idx_sn:]:
        if idx_sn < dataset_info['num_volumes'][0] + dataset_info['num_volumes'][1]:
            dataset_info['test_subjects'].append(subject)
            idx_sn += 1

    # set patch lib temp path to local storage ..
    if job_id is not None:
        dataset_info['path'][2] = dataset_info['path'][2].format(job_id)

    # set process data train/test path for dataset_info['path'][0] ..
    if trainTestFlag == 'train':
        new_traintest_folder = 'train'
        counter_traintest_folder = 'test'
    else:
        new_traintest_folder = 'test'
        counter_traintest_folder = 'train'
    curr_traintest_folder = os.path.basename(dataset_info['path'][0])

    if curr_traintest_folder != 'train' and curr_traintest_folder != 'test':
        dataset_info['path'][0] = os.path.join(dataset_info['path'][0], new_traintest_folder)
    elif curr_traintest_folder == counter_traintest_folder:
        dataset_info['path'][0] = os.path.join(os.path.dirname(dataset_info['path'][0]), new_traintest_folder)

    gen_conf['dataset_info'][train_conf['dataset']] = dataset_info

    return gen_conf, train_conf, simnorm_conf

def conf_HCP_rc_dataset(gen_conf, train_conf, simnorm_conf=None, trainTestFlag='train'):
    dataset_path = gen_conf['dataset_path']
    if 'preprocess_num' in gen_conf.keys():
        dataset_name = train_conf['dataset']+'-'+gen_conf['preprocess_num']
        if dataset_name not in gen_conf['dataset_info'].keys():
            dataset_name = train_conf['dataset']
        train_conf['patch_filename_pattern'] = gen_conf['preprocess_num'] + '-' + train_conf['patch_filename_pattern']
        if simnorm_conf is not None:
            sim_name = 'sim-' + gen_conf['preprocess_num']
            if sim_name in simnorm_conf.keys():
                simnorm_conf['sim'] = simnorm_conf['sim-'+gen_conf['preprocess_num']]
            if 'seed' in simnorm_conf['sim'].keys():
                simnorm_conf['sim']['seed'] = simnorm_conf['sim']['seed'] * int(gen_conf['preprocess_num']) + 1 # fix seed
    else:
        dataset_name = train_conf['dataset']
    job_id = gen_conf['job_id'] if 'job_id' in gen_conf.keys() else None

    dataset_info = gen_conf['dataset_info'][dataset_name]

    if 'preprocess_num' in gen_conf.keys():
        dataset_info['path'][0] = dataset_info['path'][0].format(gen_conf['preprocess_num'])

    path = dataset_info['path'][1] # original dir
    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]

    whole_dataset_path = os.path.join(dataset_path, path)
    subject_lib = sorted([os.path.basename(subject) for subject in glob.glob(whole_dataset_path+'/*')])

    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    dataset_info['training_subjects'] = []
    idx_sn = 0
    for subject in subject_lib:
        if idx_sn < dataset_info['num_volumes'][0]:
            dataset_info['training_subjects'].append(subject)
            idx_sn += 1

    dataset_info['test_subjects'] = []
    for subject in subject_lib[idx_sn:]:
        if idx_sn < dataset_info['num_volumes'][0] + dataset_info['num_volumes'][1]:
            dataset_info['test_subjects'].append(subject)
            idx_sn += 1

    # set patch lib temp path to local storage ..
    if job_id is not None:
        dataset_info['path'][2] = dataset_info['path'][2].format(job_id)

    # set process data train/test path for dataset_info['path'][0] ..
    if trainTestFlag == 'train':
        new_traintest_folder = 'train'
        counter_traintest_folder = 'test'
    else:
        new_traintest_folder = 'test'
        counter_traintest_folder = 'train'
    curr_traintest_folder = os.path.basename(dataset_info['path'][0])

    if curr_traintest_folder != 'train' and curr_traintest_folder != 'test':
        dataset_info['path'][0] = os.path.join(dataset_info['path'][0], new_traintest_folder)
    elif curr_traintest_folder == counter_traintest_folder:
        dataset_info['path'][0] = os.path.join(os.path.dirname(dataset_info['path'][0]), new_traintest_folder)

    gen_conf['dataset_info'][train_conf['dataset']] = dataset_info

    return gen_conf, train_conf, simnorm_conf

def conf_Multimodal_dataset(gen_conf, train_conf, simnorm_conf = None, trainTestFlag='train'):
    dataset_path = gen_conf['dataset_path']
    print(gen_conf['preprocess_num'])
    if 'preprocess_num' in gen_conf.keys():
        dataset_name = train_conf['dataset']+'-'+gen_conf['preprocess_num']
        if dataset_name not in gen_conf['dataset_info'].keys():
            dataset_name = train_conf['dataset']
        if 'patch_filename_pattern' in train_conf.keys():
            train_conf['patch_filename_pattern'] = gen_conf['preprocess_num'] + '-' + train_conf['patch_filename_pattern']
        if simnorm_conf is not None:
            simnorm_conf['sim'] = simnorm_conf['sim-'+gen_conf['preprocess_num']]
    else:
        dataset_name = train_conf['dataset']
    job_id = gen_conf['job_id'] if 'job_id' in gen_conf.keys() else None

    dataset_info = gen_conf['dataset_info'][dataset_name]
    path = dataset_info['path'][1] # original dir
    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]

    whole_dataset_path = os.path.join(dataset_path, path)
    subject_lib = sorted([os.path.basename(subject) for subject in glob.glob(whole_dataset_path+'/*')])

    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    dataset_info['training_subjects'] = []
    idx_sn = 0
    for subject in subject_lib:
        if idx_sn < dataset_info['num_volumes'][0]:
            dataset_info['training_subjects'].append(subject)
            idx_sn += 1

    dataset_info['test_subjects'] = []
    for subject in subject_lib[idx_sn:]:
        if idx_sn < dataset_info['num_volumes'][0] + dataset_info['num_volumes'][1]:
            dataset_info['test_subjects'].append(subject)
            idx_sn += 1

    # set patch lib temp path to local storage ..
    if job_id is not None:
        dataset_info['path'][2] = dataset_info['path'][2].format(job_id)

    # set process data train/test path for dataset_info['path'][0] ..
    if trainTestFlag == 'train':
        new_traintest_folder = 'train'
        counter_traintest_folder = 'test'
    else:
        new_traintest_folder = 'test'
        counter_traintest_folder = 'train'
    curr_traintest_folder = os.path.basename(dataset_info['path'][0])

    if curr_traintest_folder != 'train' and curr_traintest_folder != 'test':
        dataset_info['path'][0] = os.path.join(dataset_info['path'][0], new_traintest_folder)
    elif curr_traintest_folder == counter_traintest_folder:
        dataset_info['path'][0] = os.path.join(os.path.dirname(dataset_info['path'][0]), new_traintest_folder)

    gen_conf['dataset_info'][train_conf['dataset']] = dataset_info

    return gen_conf, train_conf, simnorm_conf

def save_conf_info(gen_conf, train_conf):
    dataset_name = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset_name]

    # check and create parent folder
    csv_filename_gen = generate_output_filename(
        gen_conf['log_path'],
        train_conf['dataset'],
        '_gen_conf',
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'csv')
    csv_filename_train = generate_output_filename(
        gen_conf['log_path'],
        train_conf['dataset'],
        '_train_conf', # todo: should be different if random sampling
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'csv')
    csv_filename_dataset = generate_output_filename(
        gen_conf['log_path'],
        train_conf['dataset'],
        '_dataset',
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'csv')
    ## check and make folders
    csv_foldername = os.path.dirname(csv_filename_gen)
    if not os.path.isdir(csv_foldername) :
        os.makedirs(csv_foldername)

    # save gen_conf
    with open(csv_filename_gen, 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, gen_conf.keys())
        w.writeheader()
        w.writerow(gen_conf)

    with open(csv_filename_train, 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, train_conf.keys())
        w.writeheader()
        w.writerow(train_conf)

    with open(csv_filename_dataset, 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, dataset_info.keys())
        w.writeheader()
        w.writerow(dataset_info)

def generate_output_filename(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension):
    file_pattern = '{}/{}/{}-{}-{}-{}-{}.{}'
    return file_pattern.format(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension)
