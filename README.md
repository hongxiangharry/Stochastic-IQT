# Stochastic-IQT

## Updates：
* [2023/4/30] We are still working on releasing the full version of source code for [the accepted MedIA paper](https://authors.elsevier.com/a/1g%7EwU4rfPmE0Lh), please keep watching on this page. At the moment, please refer to the pre-released IQT tutorial code described below.

## Overview

This is an official repo for the stochastic image quality transfer (IQT) project, which achieves the enhancement of the 3D low-field magnetic resonance images on (1)  global contrast and (2) resolution on slice direction. This work is under review and the source code will be released after the completion of the review process.

Low-field ($<1$T) magnetic resonance imaging (MRI) scanners remain in widespread use in low- and middle-income countries (LMICs) and are commonly used for some applications in higher income countries e.g. for small child patients with obesity, claustrophobia, implants, or tattoos. However, low-field MR images commonly have lower resolution and poorer contrast than images from high field (1.5T, 3T, and above). Here, we present Image Quality Transfer (IQT) to enhance low-field structural MRI by estimating from a low-field image the image we would have obtained from the same subject at high field. Our approach uses (i) a stochastic low-field image simulator as the forward model to capture uncertainty and variation in the contrast of low-field images corresponding to a particular high-field image, and (ii) an anisotropic U-Net variant specifically designed for the IQT inverse problem. We evaluate the proposed algorithm both in simulation and using multi-contrast (T1-weighted, T2-weighted, and fluid attenuated inversion recovery (FLAIR)) clinical low-field MRI data from an LMIC hospital. We show the efficacy of IQT in improving contrast and resolution of low-field MR images. We demonstrate that IQT-enhanced images have potential for enhancing visualisation of anatomical structures and pathological lesions of clinical relevance from the perspective of radiologists. IQT is proved to have capability of boosting the diagnostic value of low-field MRI, especially in low-resource settings. 



## Prerequisite and Software Dependency

The demo and program was tested to run on a Linux operating system (Centos or Ubuntu) with Intel i5 CPU and Nvidia Titan V GPU. To run the software, we can set up a conda environment with the following packages:

```
------------------------------------------
Name                     |Version         
-------------------------|----------------
h5py                     | 2.10.0           
hdf5                     | 1.10.5           
Keras                    | 2.3.1          
matplotlib               | 3.1.1            
numpy                    | 1.17.3          
scikit-image             | 0.16.2           
scikit-learn             | 0.22             
scipy                    | 1.3.2            
tensorboard              | 1.13.1           
tensorflow               | 1.13.1           
tensorflow-gpu           | 1.13.1          
nibabel                  | 2.1.0          
------------------------------------------
```

## Installation

### Required configuration

* Download a [Miniconda](https://docs.conda.io/en/latest/miniconda.html) version for your OS.
* Tensorflow 1.13 [requirements](https://www.tensorflow.org/install/gpu): CUDA 10.0, Nvidia GPU driver 418.x or higher, cuDNN (>=7.6). Select to download the suitable Nvidia GPU driver from [here](https://www.nvidia.com/download/index.aspx?lang=en-us).
* Download and install [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3) to view nifty files.

### Conda Environmental Set-up

On Miniconda: 
1. Create a virtual environment by `conda create -y -n iqt python=3.6.8`
2. Enter the environment by `source activate iqt`
3. Install required packages by:

~~~shell
```
pip install nibabel==2.1.0 # If any error, run `pip install nibabel==2.1.0 --user` instead.
conda install -y h5py=2.10.0 ipython=7.10.1 jupyter=1.0.0 scipy=1.3.2 scikit-image=0.16.2 scikit-learn=0.22 
pip install tensorflow==1.13.1 tensorflow-gpu==1.13.1 keras==2.3.1 # If any error, add `--user` to the end of this command line.
conda install -y cudatoolkit=10.0 cudnn pyyaml
```
~~~

### Get started (On Jupyter notebook)

1. In a linux terminal, run `git clone https://github.com/hongxiangharry/IQT_tutorial.git` to download source codes to your workspace.
2. Make sure you have run `source activate iqt`.
3. Under the workspace, launch Jupyter Notebook by `jupyter notebook`, then open `IQT.ipynb`.
4. Follow the further instruction in the notebook.
