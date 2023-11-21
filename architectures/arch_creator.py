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
""" Network architecture selection module."""

from .IsoUnet import generate_iso_unet_model
from .SRUnet import generate_srunet_model
from .AnisoUnetCustomLoss import generate_aniso_unet_model
from .AnisoUnetCustomLossUncertainty import generate_aniso_unet_model as generate_aniso_unet_uncertain_model
from .ESPCN import generate_espcn_model

def generate_model(gen_conf, train_conf) :
    approach = train_conf['approach']

    if approach == 'IsoUnet' :
        return generate_iso_unet_model(gen_conf, train_conf)
    if approach == 'SRUnet' :
        return generate_srunet_model(gen_conf, train_conf)
    if approach == 'AnisoUnet' :
        return generate_aniso_unet_model(gen_conf, train_conf)
    if approach == 'Uncertain-AnisoUnet':
        return generate_aniso_unet_uncertain_model(gen_conf, train_conf)
    if approach == 'espcn' :
        return generate_espcn_model(gen_conf, train_conf)

    return None
