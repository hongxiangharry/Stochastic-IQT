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
""" Config """

general_configuration = {
    'dataset_path' : None,
    'base_path' : None, # PC
    'job_name' : 'default', # 'srunet16_16_2_nf4' or 'anisounet16_16_2_nf4'
    'log_path_' : 'log',
    'model_path_' : 'models',
    'results_path_' : 'result',
    'evaluation_path_': 'evaluation',
    'tensorboard_path_': 'tblogs',
    'dataset_info' : {
        # 'HCP-rc': {
        #     'format': 'nii.gz',
        #     'num_volumes': [60, 10],  # train and test
        #     'modalities': 1,
        #     'general_pattern': '{}/T1w/{}_acpc_dc_restore_brain{}.nii.gz', ## Need to specify
        #     'path': ['process/HCP',       # directory to processed image data
        #              'origin/HCP',        # directory to original image data
        #              'upatch/HCP',        # directory to unzip patches [=path[3] if None]
        #              'patch/HCP'],        # directory to extract patches
        #     'in_postfix_pattern': '_sim036T_{}x{}', ## process/origin input postfix
        #     'out_postfix_pattern': '_sim036T_{}x_groundtruth', # process/origin output postfix
        #     'modality_categories': ['T1w'],
        #     'downsample_scale': 8,
        #     'sparse_scale': [1, 1, 8],
        #     'shrink_dim': 3,
        #     'is_preproc': False,     # input pre-processing
        #     'upsample_scale': [1, 1, 8],
        #     'interp_order': 3,       # try 0-5 order for interpolation if needed
        #     'is_groundtruth_masked': False,
        #     'is_sim_WM_fixed': False
        # },
        'MBB-rc': {
            'format': 'nii.gz',
            'num_volumes': [60, 10],  # train and test
            'modalities': 1,
            'general_pattern': '{}/{}{}.nii.gz', # '{}/T1w/{}_acpc_dc_restore_brain{}.nii.gz',  ## Need to specify
            'path': ['process/MBB',     # directory to processed image data, add train/test folders # to-do list
                     'origin/MBB',      # directory to original image data
                     'upatch/MBB',      # directory to unzip patches [=path[3] if None]
                     'patch/MBB'],      # directory to extract patches
            'in_postfix_pattern': '_sim036T_{}x{}',  ## process/origin input postfix
            'out_postfix_pattern': '_sim036T_{}x_groundtruth',  # process/origin output postfix
            'modality_categories': ['FLAIR'],
            'downsample_scale': 8,
            'sparse_scale': [1, 1, 8],
            'shrink_dim': 3,
            'is_preproc': False,  # input pre-processing
            'upsample_scale': [1, 1, 8],
            'interp_order': 3,  # try 0-5 order for interpolation if needed
            'is_groundtruth_masked': False,
            'is_sim_WM_fixed': False
        }
    }
}

simulation_configuration_train = {
    'sim': {
        'is_sim': False,
        'distribution': 'gaussian', # 'point' or 'gaussian' distrbution
        'params': {
            'mean': [35.031, 42.329], # WM, GM
            'cov': [[34.048, 39.850], [39.850, 50.293]]
        },
        'sim_pdf': [1],
        'slice_thickness_factor': [1, 1, 6],
        'gap': [0, 0, 2],
        'seed': 300,
        'contrast_type': 'rc'
    },
    'norm': {
        'hist_pattern': 'hist_{}_{}.npy',
        'hist_suffixes': ['lf', 'hf'],
        'normType': 'merge'
    }
}

simulation_configuration_test = {
    'sim': {
        'is_sim': False,
        'distribution': 'gaussian', # 'point' or 'gaussian' distrbution
        'params': {
            'mean': [35.031, 42.329], # WM, GM
            'cov': [[34.048, 39.850], [39.850, 50.293]]
        },
        'sim_pdf': [1],
        'slice_thickness_factor': [1, 1, 6],
        'gap': [0, 0, 2],
        'seed': 305,
        'contrast_type': 'rc'
    },
    'norm': {
        'hist_pattern': 'hist_{}_{}.npy',
        'hist_suffixes': ['lf', 'hf'],
        'normType': 'merge'
    }
}

# simulation_configuration_train = {
#     'sim': {
#         'is_sim': False,
#         'distribution': 'gaussian', # 'point' or 'gaussian' distrbution
#         'params': {
#             'mean': [64.50, 54.14], # WM, GM
#             'cov': [[78.47, 71.50],
#                     [71.50, 73.91]]
#         },
#         'sim_pdf': [1],
#         'slice_thickness_factor': [1, 1, 6],
#         'gap': [0, 0, 2],
#         'seed': 300,
#         'contrast_type': 'rc'
#     },
#     'norm': {
#         'hist_pattern': 'hist_{}_{}.npy',
#         'hist_suffixes': ['lf', 'hf'],
#         'normType': 'merge'
#     }
# }
#
# simulation_configuration_test = {
#     'sim': {
#         'is_sim': False,
#         'distribution': 'gaussian', # 'point' or 'gaussian' distrbution
#         'params': {
#             'mean': [64.50, 54.14], # WM, GM
#             'cov': [[78.47, 71.50], [71.50, 73.91]]
#         },
#         'sim_pdf': [1],
#         'slice_thickness_factor': [1, 1, 6],
#         'gap': [0, 0, 2],
#         'seed': 305,
#         'contrast_type': 'rc'
#     },
#     'norm': {
#         'hist_pattern': 'hist_{}_{}.npy',
#         'hist_suffixes': ['lf', 'hf'],
#         'normType': 'merge'
#     }
# }

training_configuration = {
    'retrain' : False,
    'patchlib-rebuild' : False,
    'activation' : 'null',
    'approach' : 'AnisoUnet', # `SRUnet` or `AnisoUnet`
    'dataset' : 'MBB-rc',
    'num_dataset' : 1,
    'dimension' : 3,
    'extraction_step' : (16, 16, 2),
    'extraction_step_test' :(16, 16, 2),
    'loss' : 'mean_squared_error',
    'metrics' : ['mse'],
    'batch_size' : 32,
    'num_train_sbj': 3,
    'num_training_patches': 25000,
    'num_epochs' : 3,
    'optimizer' : 'Adam',
    'output_shape' : (32, 32, 32),
    'output_shape_test' : (16, 16, 16),
    'patch_shape' : (32, 32, 4),
    'max_num_patches': 1750,
    'max_num_patches_per_file': 1,
    'patch_sampling_rate' : 1, # only for training
    'th_info': {
        'type': 'mean'
    },
    'bg_discard_percentage' : 0.2,
    'patience' : 1000,
    'validation_split' : 0.20,
    'use_multiprocessing': False,
    'workers': 4,
    'verbose' : 1, # 0: save message flow in log, 1: process bar record, 2: epoch output record
    'shuffle' : True,
    'decay' : 0.000001,
    'learning_rate' : 0.001,
    'downsize_factor' : 1,
    'dropout_rate' : 0.5,
    'regu_param' : 0.001,
    'num_kernels' : 2,
    'num_filters' : 16,
    'mapping_times' : 2,
    'num_levels': 5,
    'ishomo': False,
    'cases' : 1, # number of training cases
    'patch_filename_pattern': '{}-{}-{}-{}.{}',
}

test_configuration = {
    'ensemble_type': 'ave|rc',
    'retrain' : False,
    'activation' : 'null',
    'approach' : 'AnisoUnet', # `SRUnet` or `AnisoUnet`
    'dataset' : 'MBB-rc',
    'num_dataset' : 1,
    'dimension' : 3,
    'extraction_step' : (16, 16, 2),
    'extraction_step_test' :(16, 16, 2),
    'loss' : 'mean_squared_error',
    'metrics' : ['mse'],
    'batch_size' : 32,
    'num_epochs' : 100,
    'optimizer' : 'Adam',
    'output_shape' : (32, 32, 32),
    'output_shape_test' : (16, 16, 16),
    'patch_shape' : (32, 32, 4),
    'bg_discard_percentage' : 0.2,
    'validation_split' : 0.20,
    'use_multiprocessing': False,
    'workers': 4,
    'verbose' : 1, # 0: save message flow in log, 1: process bar record, 2: epoch output record
    'shuffle' : True,
    'decay' : 0.000001,
    'learning_rate' : 0.001,
    'downsize_factor' : 1,
    'dropout_rate': 0.5,
    'regu_param': 0.001,
    'num_kernels' : 2,
    'num_filters' : 16,
    'mapping_times' : 2,
    'num_levels': 5,
    'ishomo': False,
    'is_selfnormalised': False,
    'cases' : 1, # number of test cases
    'patch_filename_pattern': '{}-{}-{}-{}.{}',
}
