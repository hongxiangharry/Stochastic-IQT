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
""" Evaluate the performance of the IQT model: calculate the image quality metrics and save to a csv file. """

import csv
import sys
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim # for new skimage > v.16
from sklearn.metrics import r2_score
from scipy import stats
import nibabel as nib
from utils.ioutils import read_dataset, read_result_volume, read_masks, read_data_path, read_result_volume_path, read_volume

def interp_evaluation(gen_conf, test_conf, case_name = 1):
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_info['sparse_scale'] = [1, 1, 1]
    dataset_info['in_postfix'] = dataset_info['interp_postfix']

    im_interp, im_gt = read_dataset(gen_conf, test_conf, 'test') # shape : (subject, modality, mri_shape)
    compare_images_and_get_stats(gen_conf, test_conf, im_gt, im_interp, case_name, save_filename='interp')
    return True

def image_evaluation_lowcost_uncertainty(gen_conf, test_conf, i_dataset = None, originProcessFlag='origin'):
    '''
    load img/std path for uncertainty estimation
    '''
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    sparse_scale = dataset_info['sparse_scale']
    if 'is_upsample' in dataset_info.keys():
        is_upsample = dataset_info['is_upsample']
    else:
        is_upsample = False

    # load the ground truth image and result
    if sparse_scale == [1, 1, 1] and is_upsample is False:
        im_interp_paths, im_gt_paths = read_data_path(gen_conf, test_conf, 'eval', originProcessFlag=originProcessFlag) # shape : (subject, modality, mri_shape)
    else:
        _, im_gt_paths = read_data_path(gen_conf, test_conf, 'eval', originProcessFlag=originProcessFlag)  # shape : (subject, modality, mri_shape)
    # load image recon path templates: for img/std
    im_recon_paths = read_result_volume_path(gen_conf, test_conf, str(i_dataset)+'{}') # shape : (subject, modality, mri_shape)

    # load mask if provided:
    if 'mask_postfix' in dataset_info.keys():
        masks = read_masks(gen_conf, test_conf, 'test')  # shape : (subject, modality, mri_shape)
        print(masks.dtype, masks.shape)
    else:
        masks = None

    # compare image and get stats
    get_accuracy_uncertainty_metrics_lowcost(gen_conf, test_conf, im_gt_paths, im_recon_paths, i_dataset, save_filename='recon', masks=masks)
    if sparse_scale == [1, 1, 1] and is_upsample is False:
        get_accuracy_uncertainty_metrics_lowcost(gen_conf, test_conf, im_gt_paths, im_interp_paths, i_dataset, save_filename='interp')

def image_evaluation_lowcost(gen_conf, test_conf, i_dataset = None, originProcessFlag='origin'):
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    sparse_scale = dataset_info['sparse_scale']
    if 'is_upsample' in dataset_info.keys():
        is_upsample = dataset_info['is_upsample']
    else:
        is_upsample = False

    # load the ground truth image and result
    if sparse_scale == [1, 1, 1] and is_upsample is False:
        im_interp_paths, im_gt_paths = read_data_path(gen_conf, test_conf, 'eval', originProcessFlag=originProcessFlag) # shape : (subject, modality, mri_shape)
    else:
        _, im_gt_paths = read_data_path(gen_conf, test_conf, 'eval', originProcessFlag=originProcessFlag)  # shape : (subject, modality, mri_shape)
    im_recon_paths = read_result_volume_path(gen_conf, test_conf, i_dataset) # shape : (subject, modality, mri_shape)

    # load mask if provided:
    if 'mask_postfix' in dataset_info.keys():
        masks = read_masks(gen_conf, test_conf, 'test')  # shape : (subject, modality, mri_shape)
        print(masks.dtype, masks.shape)
    else:
        masks = None

    # compare image and get stats
    compare_images_and_get_stats_lowcost(gen_conf, test_conf, im_gt_paths, im_recon_paths, i_dataset, save_filename='recon', masks=masks)
    if sparse_scale == [1, 1, 1] and is_upsample is False:
        compare_images_and_get_stats_lowcost(gen_conf, test_conf, im_gt_paths, im_interp_paths, i_dataset, save_filename='interp')

def get_accuracy_uncertainty_metrics_lowcost(gen_conf, test_conf, im_gt_paths, im_recon_paths, case_name = None, save_filename='ind', masks = None):
    '''
    load mean/std outputs, output uncertainty metrics by subjects
    '''
    if case_name is None:
        case_name = 'all'

    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    subject_lib = dataset_info['test_subjects']
    modality_categories = dataset_info['modality_categories']
    num_volumes = dataset_info['num_volumes'][1]
    modalities = dataset_info['modalities']

    csv_file = generate_output_filename(
        gen_conf['evaluation_path'],
        test_conf['dataset'],
        'stat-'+save_filename+'-c'+str(case_name),
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        'csv')

    csv_folderpath = os.path.dirname(csv_file)
    if not os.path.isdir(csv_folderpath) :
        os.makedirs(csv_folderpath)
    headers = ['subject', 'modality', 'RMSE', 'NRMSE', 'Median', 'PSNR', 'MSSIM', 'CORR', 'R2', 'S2', 'NLL']

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            im_gt = read_volume(im_gt_paths[img_idx * modalities + mod_idx])
            im_recon = read_volume(im_recon_paths[img_idx * modalities + mod_idx].format(''))
            im_std = read_volume(im_recon_paths[img_idx * modalities + mod_idx].format('_std'))
            if masks is None:
                mask = (im_gt != 0)
            else:
                mask = masks[img_idx, mod_idx]
            # compare image and get stats
            m, nm, m2, p, s, corr, r2, s2, nll = _compare_images_and_get_uncertain_stats(im_gt,
                                                            im_recon,
                                                            im_std,
                                                            mask,
                                                            "whole: image no: {}, modality no: {}".format(img_idx, mod_idx))
            # save statistics
            stats = [m, nm, m2, p, s, corr, r2, s2, nll]
            # Save the stats to a CSV file:
            save_stats(csv_file, subject_lib[img_idx], modality_categories[mod_idx], headers, stats)

def compare_images_and_get_stats_lowcost(gen_conf, test_conf, im_gt_paths, im_recon_paths, case_name = None, save_filename='ind', masks = None):
    if case_name is None:
        case_name = 'all'

    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    subject_lib = dataset_info['test_subjects']
    modality_categories = dataset_info['modality_categories']
    num_volumes = dataset_info['num_volumes'][1]
    modalities = dataset_info['modalities']

    csv_file = generate_output_filename(
        gen_conf['evaluation_path'],
        test_conf['dataset'],
        'stat-'+save_filename+'-c'+str(case_name),
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        'csv')

    csv_folderpath = os.path.dirname(csv_file)
    if not os.path.isdir(csv_folderpath) :
        os.makedirs(csv_folderpath)
    headers = ['subject', 'modality', 'RMSE', 'NRMSE', 'Median', 'PSNR', 'MSSIM']

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            im_gt = read_volume(im_gt_paths[img_idx * modalities + mod_idx])
            im_recon = read_volume(im_recon_paths[img_idx * modalities + mod_idx])
            if masks is None:
                mask = (im_gt != 0)
            else:
                mask = masks[img_idx, mod_idx]
            # compare image and get stats
            m, nm, m2, p, s = _compare_images_and_get_stats(im_gt,
                                                            im_recon,
                                                            mask,
                                                            "whole: image no: {}, modality no: {}".format(img_idx,
                                                                                                          mod_idx))
            # save statistics
            stats = [m, nm, m2, p, s]
            # Save the stats to a CSV file:
            save_stats(csv_file, subject_lib[img_idx], modality_categories[mod_idx], headers, stats)

'''
    - read ground truth
    - read reconstructed image
    - compare image and get stats
    - save statistics
    - compare difference map
'''
def image_evaluation(gen_conf, test_conf, i_dataset = None):
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    sparse_scale = dataset_info['sparse_scale']
    is_upsample = dataset_info['is_upsample']

    if sparse_scale == [1, 1, 1] and is_upsample is False:
        im_interp, im_gt = read_dataset(gen_conf, test_conf, 'test') # shape : (subject, modality, mri_shape)
    else:
        _, im_gt = read_dataset(gen_conf, test_conf, 'test')  # shape : (subject, modality, mri_shape)
    im_recon = read_result_volume(gen_conf, test_conf, i_dataset) # shape : (subject, modality, mri_shape)

    # load mask if provided:
    if 'out_postfix' in dataset_info.keys():
        masks = read_masks(gen_conf, test_conf, 'test')  # shape : (subject, modality, mri_shape)
        print(masks.dtype, masks.shape)
    else:
        masks = None

    # compare image and get stats
    compare_images_and_get_stats(gen_conf, test_conf, im_gt, im_recon, i_dataset, save_filename='recon', masks=masks)
    if sparse_scale == [1, 1, 1] and is_upsample is False:
        compare_images_and_get_stats(gen_conf, test_conf, im_gt, im_interp, i_dataset, save_filename='interp')


def compare_images_and_get_stats(gen_conf, test_conf, im_gt, im_recon, case_name = None, save_filename='ind', masks = None):
    if case_name is None:
        case_name = 'all'

    num_volumes = im_gt.shape[0]
    modalities = im_gt.shape[1]

    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    subject_lib = dataset_info['test_subjects']
    modality_categories = dataset_info['modality_categories']
    csv_file = generate_output_filename(
        gen_conf['evaluation_path'],
        test_conf['dataset'],
        'stat-'+save_filename+'-c'+str(case_name),
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        'csv')

    csv_folderpath = os.path.dirname(csv_file)
    if not os.path.isdir(csv_folderpath) :
        os.makedirs(csv_folderpath)
    headers = ['subject', 'modality', 'RMSE', 'NRMSE', 'Median', 'PSNR', 'MSSIM']

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            if masks is None:
                mask = (im_gt[img_idx, mod_idx] != 0)
            else:
                mask = masks[img_idx, mod_idx]
            # compare image and get stats
            m, nm, m2, p, s = _compare_images_and_get_stats(im_gt[img_idx, mod_idx],
                                                            im_recon[img_idx, mod_idx],
                                                            mask,
                                                            "whole: image no: {}, modality no: {}".format(img_idx,
                                                                                                          mod_idx))
            # save statistics
            stats = [m, nm, m2, p, s]
            # Save the stats to a CSV file:
            save_stats(csv_file, subject_lib[img_idx], modality_categories[mod_idx], headers, stats)


def save_stats(csv_file, subject, modality, headers, stats):
    """
    Args:
        csv_file (str) : the whole path to the csv file
        subject (str): subject ID
        modality (str): modality name
        headers (list): list of metrics e.g. ['subject name', 'rmse ', 'median', 'psnr', 'mssim']
        stats (list): the errors for the corresponding subject e.g [1,2,3,4]

    """
    # if csv file exists, just update with the new entries:
    assert len(headers) == len([subject] + [modality] + stats)

    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            r = csv.reader(f)
            rows = list(r)
            rows_new = []
            new_row_flag = False ## if subject and modality is not existed in csv
            # copy the old table and update if necessary:
            for row in rows:
                if row[0] == subject and row[1] == modality: # update for the corresponding subject and modality
                    rows_new.append([subject]+[modality]+stats)
                    new_row_flag = True
                else:
                    rows_new.append(row)

            # add the new entry if it does not exist in the old table.
            # if not([subject, modality] in [[row for row in rows_new][0:2]]):
            if new_row_flag == False:
                rows_new.append([subject] + [modality]+ stats)
    else:
        rows_new = [headers, [subject]+ [modality]+stats]

    # save it to a csv file:
    with open(csv_file, 'w') as g:
        w = csv.writer(g)
        for row in rows_new:
            w.writerow(row)

def _compare_images_and_get_uncertain_stats(img_gt, img_est, img_std, mask, name=''):
    """Compute RMSE, PSNR, MSSIM:
    Args:
         img_gt: (3D numpy array )
         ground truth volume
         img_est: (3D numpy array) predicted volume
         mask: (3D array) the mask whose the tissue voxels
         are labelled as 1 and the rest as 0
     Returns:
         m : RMSE
         m2: median of voxelwise RMSE
         p: PSNR
         s: MSSIM
     """
    blockPrint()
    m = compute_rmse(img_gt, img_est, mask)
    nm= compute_nrmse(img_gt, img_est, mask, normal_type ='mean')
    m2= compute_rmse_median(img_gt, img_est, mask)
    p = compute_psnr(img_gt, img_est, mask)
    s = compute_mssim(img_gt, img_est, mask)
    corr = compute_pearson_corr(img_gt, img_est, mask)
    r2 = compute_r2_score(img_gt, img_est, mask)
    s2 = compute_sharpness_square(img_std, mask)
    nll = compute_nll(img_est, img_std, img_gt, mask, scaled=True)

    enablePrint()

    print("Errors (%s)"
          "\nRMSE: %.10f \nNRMSE: %.10f \nMedian: %.10f "
          "\nPSNR: %.6f \nSSIM: %.6f \n Pearson Corr: %.6f"
          "\nR2: %.6f \nSharpness2: %.10f \nNLL: %.10f" % (name, m, nm, m2, p, s, corr, r2, s2, nll))

    return m, nm, m2, p, s, corr, r2, s2, nll

def _compare_images_and_get_stats(img_gt, img_est, mask, name=''):
    """Compute RMSE, PSNR, MSSIM:
    Args:
         img_gt: (3D numpy array )
         ground truth volume
         img_est: (3D numpy array) predicted volume
         mask: (3D array) the mask whose the tissue voxels
         are labelled as 1 and the rest as 0
     Returns:
         m : RMSE
         m2: median of voxelwise RMSE
         p: PSNR
         s: MSSIM
     """
    blockPrint()
    m = compute_rmse(img_gt, img_est, mask)
    nm= compute_nrmse(img_gt, img_est, mask, normal_type ='mean')
    m2= compute_rmse_median(img_gt, img_est, mask)
    p = compute_psnr(img_gt, img_est, mask)
    s = compute_mssim(img_gt, img_est, mask)
    enablePrint()

    print("Errors (%s)"
          "\nRMSE: %.10f \nNRMSE: %.10f \nMedian: %.10f "
          "\nPSNR: %.6f \nSSIM: %.6f" % (name, m, nm, m2, p, s))

    return m, nm, m2, p, s


def compute_rmse(img1, img2, mask):
    if img1.shape != img2.shape:
        print("shape of img1 and img2: %s and %s" % (img1.shape, img2.shape))
        raise ValueError("the size of img 1 and img 2 do not match")
    mse = np.sum(((img1-img2)**2)*mask) \
          /(mask.sum())
    return np.sqrt(mse)

def compute_nrmse(img1, img2, mask, normal_type ='mean'):
    '''
        two ways to express normalised RMSE(nrmse): https://en.wikipedia.org/wiki/Root-mean-square_deviation
    '''
    if img1.shape != img2.shape:
        print("shape of img and img_true: %s and %s" % (img1.shape, img2.shape))
        raise ValueError("the size of img and img_true do not match")
    rmse = compute_rmse(img1, img2, mask)
    if normal_type == 'mean':
        norm = np.sum(img1 * mask) / (mask.sum())
    elif normal_type == 'range':
        true_min, true_max = np.min(img1[mask]), np.max(img1)
        if true_min >= 0:
            norm = true_max
        else:
            norm = true_max - true_min
    nrmse = rmse/norm
    return nrmse

def compute_rmse_median(img1, img2, mask):
    if img1.shape != img2.shape:
        print("shape of img1 and img2: %s and %s" % (img1.shape, img2.shape))
        raise ValueError("the size of img 1 and img 2 do not match")

    # compute the voxel-wise average error:
    rmse_vol = np.sqrt((img1 - img2) ** 2 * mask)

    return np.median(rmse_vol[rmse_vol!=0])


def compute_mssim(img1, img2, mask, volume=False):
    if img1.shape != img2.shape:
        print("shape of img1 and img2: %s and %s" % (img1.shape, img2.shape))
        raise ValueError("the size of img 1 and img 2 do not match")
    img1=img1*mask
    img2=img2*mask

    m, S = ssim(img1,img2,
                dynamic_range=np.max(img1[mask])-np.min(img1[mask]),
                gaussian_weights=True,
                sigma= 2.5, #1.5,
                use_sample_covariance=False,
                full=True,
                multichannel=True)
    if volume:
        return S * mask
    else:
        mssim = np.sum(S * mask) / (mask.sum())
        return mssim


def compute_psnr(img1, img2, mask):
    """ Compute PSNR
    Arg:
        img1: ground truth image
        img2: test image
    """
    if img1.shape != img2.shape:
        print("shape of img1 and img2: %s and %s" % (img1.shape, img2.shape))
        raise ValueError("the size of img 1 and img 2 do not match")

    img1 = img1 * mask
    img2 = img2 * mask

    true_min, true_max = np.min(img1[mask]), np.max(img1)

    if true_min >= 0:
        # most common case (255 for uint8, 1 for float)
        dynamic_range = true_max
    else:
        dynamic_range = true_max - true_min

    rmse = compute_rmse(img1, img2, mask)
    return 10 * np.log10((dynamic_range ** 2) / (rmse**2))

def compute_differencemaps_t1t2(img_gt, img_est, mask, outputfile, no_channels,
                           save_as_ijk=True, gt_dir=None, gt_header=None, category=None):

    # Compute the L2 deviation and SSIM:
    rmse_volume = np.sqrt(((img_gt - img_est) ** 2)* mask[..., np.newaxis])
    blockPrint()
    ssim_volume = compute_mssim(img_gt, img_est, mask, volume=True)
    enablePrint()

    # Save the error maps:
    save_dir, file_name = os.path.split(outputfile)
    header, ext = os.path.splitext(file_name)

    for k in range(no_channels):
        if not (save_as_ijk):
            print("Fetching affine transform and header from GT.")
            if no_channels > 7:
                gt_file = gt_header + '%02i.nii' % (k + 1,)
                dt_gt = nib.load(os.path.join(gt_dir, gt_file))
            else:
                dt_gt = nib.load(os.path.join(gt_dir, gt_header + str(k + 1) + '.nii'))

            affine = dt_gt.get_affine()  # fetch its affine transfomation
            nii_header = dt_gt.get_header()  # fetch its header
            img_1 = nib.Nifti1Image(rmse_volume[:, :, :, k], affine=affine, header=nii_header)
            img_2 = nib.Nifti1Image(ssim_volume[:, :, :, k], affine=affine, header=nii_header)
        else:
            img_1 = nib.Nifti1Image(rmse_volume[:, :, :, k], np.eye(4))
            img_2 = nib.Nifti1Image(ssim_volume[:, :, :, k], np.eye(4))

        print('... saving the error (RMSE) and SSIM map for ' + str(k + 1) + ' th T1/T2 element')
        nib.save(img_1, os.path.join(save_dir, '_error_' + header + '.nii')) ## todo: no enough!
        nib.save(img_2, os.path.join(save_dir, '_ssim_' + header + '.nii'))  ## todo: no enough!


####### correlation related #######
def compute_pearson_corr(img1, img2, mask):
    '''
    :param img1: -> into 1D vector
    :param img2: -> into 1D vector
    :param mask:
    :return:
    note: np.ravel() has same function as np.flatten() but less memory
    '''
    if img1.shape != img2.shape:
        print("shape of img1 and img2: %s and %s" % (img1.shape, img2.shape))
        raise ValueError("the size of img 1 and img 2 do not match")

    img1 = img1.ravel()[np.flatnonzero(mask)]
    img2 = img2.ravel()[np.flatnonzero(mask)]

    corr = np.corrcoef(img1, img2)[0,1]
    return corr

def compute_r2_score(img1, img2, mask):
    '''

    :param img1: -> into 1D vector
    :param img2: -> into 1D vector
    :param mask:
    :return:
    '''
    if img1.shape != img2.shape:
        print("shape of img1 and img2: %s and %s" % (img1.shape, img2.shape))
        raise ValueError("the size of img 1 and img 2 do not match")

    # img1 = img1 * mask
    img1 = img1.ravel()[np.flatnonzero(mask)]
    # img2 = img2 * mask
    img2 = img2.ravel()[np.flatnonzero(mask)]

    r2 = r2_score(img1, img2)
    return r2

####### uncertainty ###########
## sharpness square:
def compute_sharpness_square(img_std, mask):
    '''
    sharp2 = mean(img_std**2)
    :param img_std:
    :param mask:
    :return:
    '''
    # img_std = img_std*mask
    img_std = img_std.ravel()[np.flatnonzero(mask)]
    sharp2 = np.mean(img_std**2)

    return sharp2

## negative log likelihood
def compute_nll(img_pred, img_std, img_true, mask, scaled=True):
    '''
    NLL = -1/N sum_i^N sum_k^K log( N(y^pred-y^true, std^2) )
    :param img_pred:
    :param img_std:
    :param img_true:
    :param mask:
    :param scaled: Whether to scale the negative log likelihood by size of held out set.
    :return:

    bug history: 11/19, #residuals != img_std
    '''

    # Set residuals
    residuals = img_pred - img_true
    residuals = residuals.ravel()[np.flatnonzero(mask)]
    img_std = img_std.ravel()[np.flatnonzero(mask)]

    # Compute nll
    nll_list = stats.norm.logpdf(residuals, scale=img_std)
    nll = -1 * np.sum(nll_list)

    # Potentially scale so that sum becomes mean
    if scaled:
        nll = nll / len(nll_list)

    return nll

####### Misc ######

# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def generate_output_filename(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension):
    file_pattern = '{}/{}/{}-{}-{}-{}-{}.{}'
    return file_pattern.format(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension)