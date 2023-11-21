<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



# Low to High Field Image Qualitiy Transfer for MRI

[![MIT License][license-shield]][license-url]

This repo is an implementation of [Low-field magnetic resonance image enhancement via stochastic image quality transfer](https://www.sciencedirect.com/science/article/pii/S1361841523000683) at Medical Image Analysis (2023) with Tensorflow 2.x. 

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#update">Update</a>
    </li>
    <li>
      <a href="#overview">Overview</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites-for-computer-system">Prerequisites for Computer System</a></li>
        <li>
          <a href="#installation">Installation</a>
          <ul>
            <li><a href="#required-configuration">Required configuration</a></li>
            <li><a href="#set-up-conda-environment">Set up Conda Environment</a></li>
            <li><a href="#set-up-project-directory">Set up Project Directory</a></li>
          </ul>
        </li>
        <li><a href="#data-description-and-pre-processing">Data Description and Pre-processing</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#overall-workflow">Overall Workflow</a></li>
        <li><a href="#run-all-[train-and-test]">Run-all [Train and Test]</a></li>
        <li><a href="#deploy-fully-trained-models-[test-only]">Deploy fully-trained models [Test only]</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>




## Update

[21/11/2023] The source code v1.0 is now available to public. Please feel free to fork and give us a star.


<!-- Overview -->

## Overview

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

This is an official repo for the stochastic image quality transfer (SIQT) project, we release the full version of source code with [the accepted MedIA paper](https://authors.elsevier.com/a/1g~wU4rfPmE0Lh). This program achieves the enhancement of the 3D low-field magnetic resonance images on (1) global contrast and (2) resolution on slice direction.



Low-field (<1​T) magnetic resonance imaging (MRI) scanners remain in widespread use in low- and middle-income countries (LMICs) and are commonly used for some applications in higher income countries e.g. for small child patients with obesity, claustrophobia, implants, or tattoos. However, low-field MR images commonly have lower resolution and poorer contrast than images from high field (1.5T, 3T, and above). Here, we present Image Quality Transfer (IQT) to enhance low-field structural MRI by estimating from a low-field image the image we would have obtained from the same subject at high field. Our approach uses (i) a stochastic low-field image simulator as the forward model to capture uncertainty and variation in the contrast of low-field images corresponding to a particular high-field image, and (ii) an anisotropic U-Net variant specifically designed for the IQT inverse problem. We evaluate the proposed algorithm both in simulation and using multi-contrast (T1-weighted, T2-weighted, and fluid attenuated inversion recovery (FLAIR)) clinical low-field MRI data from an LMIC hospital. We show the efficacy of IQT in improving contrast and resolution of low-field MR images. We demonstrate that IQT-enhanced images have potential for enhancing visualisation of anatomical structures and pathological lesions of clinical relevance from the perspective of radiologists. IQT is proved to have capability of boosting the diagnostic value of low-field MRI, especially in low-resource settings.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Citation -->

## Citation

Please do cite our paper and give us a star on this git page if you feel this code useful to your research.

```
@article{lin2023low,
  title={Low-field magnetic resonance image enhancement via stochastic image quality transfer},
  author={Lin, Hongxiang and Figini, Matteo and D’Arco, Felice and Ogbole, Godwin and Tanno, Ryutaro and Blumberg, {Stefano B} and Ronan, Lisa and Brown, {Biobele J} and Carmichael, {David W} and Lagunju, Ikeoluwa and  Fernandez-Reyes, Delmiro and Alexander, {Daniel C}},
  journal={Medical Image Analysis},
  volume={87},
  pages={102807},
  year={2023},
  publisher={Elsevier}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.



### Prerequisites for Computer System

The demo and program should run on a Linux operating system (OS) with a large RAM and an Nvidia GPU with a large GRAM if using GPU mode. The following exemplifies as one of specs that we tested on a local workstation or a server:

* A Linux OS of Centos 7.9.
* An Nvidia GPU of Nvidia V100.
* Approximately 100 GB free harddisk space for training. (Relaxable if using fewer training data. ) 

This program is built on Python and runs on Conda Environment.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation

#### Required configuration

- Download a [Miniconda](https://docs.conda.io/en/latest/miniconda.html) version for your OS.

  - Command line installation for 64-bit Linux OS. Replace <path/to/directory> with the custom installation path for Miniconda:

    ```shell
    # Remember to replace <path/to/directory> with your custom path.
    mkdir -p <path/to/directory>/miniconda3
    cd <path/to/directory>
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
    bash miniconda3/miniconda.sh -bfp <path/to/directory>/miniconda3/
    rm -rf miniconda3/miniconda.sh
    ```

- Identify Tensorflow dependency:

  - As an example, we use `Tensorflow=2.4.0` and recommend to install by `conda`. The required CUDA Toolkit version is `cudatoolkit=11.0` and the corresponding cuDNN (`cudnn`) sdk version will be automatically calculated on `conda`.
    - If encountering `OSError: pydot failed to call GraphViz.` need to install [python-graphviz](https://stackoverflow.com/questions/60151961/pydot-failed-to-call-graphviz-please-install-graphviz-and-ensure-that-its-exec) in the conda environment.
  - Install Nvidia driver `450.80.02` compatible to CUDA version `cudatoolkit=11.0`; see [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#forward-compatible-upgrade).

- Download and install Standalone `SPM12` software [here](https://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/) for data pre-processing.

  - Prior to this one should donload [MATLAB Runtime 2019b](https://uk.mathworks.com/products/compiler/matlab-runtime.html) and follow the installation guideline [here](https://uk.mathworks.com/help/compiler/install-the-matlab-runtime.html).

- Download and install `ITK-SNAP 4.x` [here](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP4) for viewing nifty files.

  <p align="right">(<a href="#readme-top">back to top</a>)</p>


#### Set up Conda Environment

On Miniconda:

1. Create a virtual environment by `conda create -y -n iqt python=3.7`
2. Enter the environment by `source activate iqt`
3. Install required packages by:

```shell
conda install -y -c conda-forge cudatoolkit=11.8.0 numpy scikit-learn pandas scikit-image scipy matplotlib pydot graphviz python-graphviz
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
pip install "intensity-normalization[ants]"==1.4.5 nibabel==3.1.1
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Set up Project Directory

1. Clone the SIQT source code from Github to the local path `<path/to/project/directory>`:

   ```
   cd <path/to/project/directory>
   git clone https://github.com/hongxiangharry/Stochastic-IQT.git
   ```

2. View and put all required datasets into the following directories. <auto-gen> means the program will generate it while running.

```
SIQT_release/
├── architecture             # Neural network files
├── config               		 # Config file including the arguments for main.py
├── dataloader               # Dataloader to optimizer in terms of training data
├── utils                    # Tools and utilities
├── workflow                 # Contain of SIQT modules (workspace/simulation/training/test/evaluation)
├── LICENSE									 # MIT license
├── main.py									 # Main function
├── README.md                # Guideline of the SIQT project and program
├── data										 # Organization of data structures for training/test
		├── origin							 # MRI data downloaded and pre-processed outside the SIQT program
		├── process							 # <auto-gen> Processed MRI data after simulation and normalization
		├── patch								 # <auto-gen> Extracting patches from processed MRI data and zipping to a package
		└── upatch							 # <auto-gen> The folder to unzip the patch package
└── <default>								 # <auto-gen> The outputs for SIQT
		├── origin							 # <auto-gen> MRI data downloaded and pre-processed outside the SIQT program
		├── process							 # <auto-gen> Processed MRI data after simulation and normalization
		├── patch								 # <auto-gen> Extracting patches from processed MRI data and zipping to a package
		└── upatch							 # <auto-gen> The folder to unzip the patch package
```

3. Edit `config/config.py` and customize your own project configuration. The default arguments are set for the example model we provided below.

4. Move the non-processing MRI dataset into the  folder `data/origin`. Remember to define `general_configuration['dataset_info'][dataset]['general_pattern']` in `config/config.py` .

   <p align="right">(<a href="#readme-top">back to top</a>)</p>


### Data Description and Pre-processing

In this study, three datasets of multiple MRI modalities are adapted for each SIQT model. We use publicly available high-field structural MRI data for model training and technical evaluation, as  well as the low-field in-house data of several normal and pathological brain scans from UCH Ibadan in Nigeria. 

T1w/T2w:

The famous `HCP Wu-Minn dataset` is used for T1w/T2w image enhancement. Please follow its use term and download data [here](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) on their official website. Particularly for training, as an example, we perform additional steps for brain tissue segmentation using `SPM12`. It generates the segmentation masks for the input brain image, namely gray matter (GM), white matter (WM) and cerebral spinal fluid (CSF) , whose file names has the prefixes of 'c1', 'c2' and 'c3' to the input image file name. These steps can provided *skull-striped* data.

FLAIR:

The famous `Max Planck Institute LEMON dataset` or `MBB` is used for FLAIR image enhancement. Please follow their use term and download data [here](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html) on their official website. Same as the HCP Wu-Minn dataset, we use `SPM12` to generate the segmentation masks for GM, WM, and CSF, whose file names has the prefixes of 'c1', 'c2' and 'c3' to the input image file name. 

Therefore, each subject folder used for training SIQT should contain:

* An image file, e.g. 'T1brain.nii.gz';
* A GM mask file heading 'c1' on top of the image file name, e.g. 'c1T1brain.nii.gz';
* A WM mask file heading 'c2' on top of the image file name, e.g. 'c2T1brain.nii.gz'
* A CSF mask file heading 'c3' on top of the image file name, e.g. 'c3T1brain.nii.gz'.

We don't have permission to share in-house patient data (0.36T/1.5T) from UCH Ibadan, Nigeria.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

### Overall Workflow

A figure to be clarified... SOON

### Run-all [Train and Test]

1. Activate `iqt` environment by the command line `source activate iqt`
2. Go to the project directory: `cd <path/to/project/directory>`
3. Run with `python main.py -jn <JOB_NAME>`. The outputs will appeal in `SIQT_release/<JOB_NAME>`.

Note A: To understand the arguments in this program, please run `python main.py --help`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Deploy fully-trained models [Test only]

We provide the following trained models for those who consider evaluating SIQT directly. Please download and unzip to `SIQT_release/<JOB_NAME>/model`. Please make sure that the test low-field (0.36T) data should be skull-striped.

After all, please run `python main.py --test -jn <JOB_NAME>`.

| Modality | Data, #subjects | Upsampling r | PSNR  | SSIM  | Model Link   |
| :------: | --------------- | ------------ | ----- | ----- | ------------ |
|   T1w    | HCP, 60         | 4            | 32.66 | 0.875 | [OneDrive]() |
|   T1w    | HCP, 60         | 8            | 29.69 | 0.777 | [OneDrive]() |
|   T2w    | HCP, 60         | 4            |       |       | [OneDrive]() |
|   T2w    | HCP, 60         | 8            |       |       | [OneDrive]() |
|  FLAIR   | LEMON (MBB), 60 | 4            |       |       | [OneDrive]() |
|  FLAIR   | LEMON (MBB), 60 | 8            |       |       | [OneDrive]() |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->

## Contact

- Software and main issues: Hongxiang (Harry) Lin - @Github: hongxiangharry

- Data and model related issues: Matteo Figini - @Github

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

We thank [Ryutaro Tanno](https://rt416.github.io/) @Google DeepMind and [Stefano B Blumberg](https://www.linkedin.com/in/stefano-blumberg-5a8505244?originalSubdomain=uk) @UCL for providing us with their excellent thoughts on software engineering. Also thank Dr. Enrico Kaden for assisting on data processing at the beginning of this project. The code was inspired by Ryutaro Tanno's source code and Jose Bernal et al's segmentation code [here](https://ieeexplore.ieee.org/abstract/document/8754679).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt