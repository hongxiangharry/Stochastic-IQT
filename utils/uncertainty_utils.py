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
# This part is a derivation of Masksembles2D to 3D, whose original code follows MIT license.
# @Original Git Repo: https://github.com/nikitadurasov/masksembles
# @Citation: `Masksembles for Uncertainty Estimation`, Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua. CVPR 2021.
""" Anisotropic U-Net models with uncertainty estimation. We use a modification of Masksembles for deep ensembles implementation. """

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

class Masksembles3D(tf.keras.layers.Layer):
    """
    :class:`Masksembles2D` is high-level class that implements Masksembles approach
    for 2-dimensional inputs (similar to :class:`tensorflow.keras.layers.SpatialDropout1D`).
    :param n: int, number of masks
    :param scale: float, scale parameter similar to *S* in [1]. Larger values decrease \
        subnetworks correlations but at the same time decrease capacity of every individual model.
    Shape:
        * Input: (N, H, W, D, C) for channels last or (N, C, H, W, D) for channels first
        * Output: same shape as input
    Examples:
        m = Masksembles3D(4, 2.0)
        inputs = tf.ones([4, 28, 28, 28, 16])
        output = m(inputs)
    References:
    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua
    [2] `Batchensembles`
    """

    def __init__(self, n: int, scale: float, image_data_format: str = None):
        super(Masksembles3D, self).__init__()

        self.n = n
        self.scale = scale
        if image_data_format is None:
            self.image_data_format = K.image_data_format()
        else:
            self.image_data_format = image_data_format

    def build(self, input_shape):
        if self.image_data_format == 'channels_first':
            channels = input_shape[1]
        elif self.image_data_format == 'channels_last':
            channels = input_shape[-1]
        masks = generation_wrapper(channels, self.n, self.scale)
        self.masks = self.add_weight("masks",
                                     shape=masks.shape,
                                     trainable=False,
                                     dtype="float32")
        self.masks.assign(masks)

    def call(self, inputs, training=False):
        # inputs : [N, H, W, D, C] or [N, C, H, W, D]
        # masks : [M, C]
        x = tf.stack(tf.split(inputs, self.n))  # Return a list of split stack
        print(x.shape)
        # x : [M, N // M, H, W, D, C] or [M, N // M, C, H, W, D]
        # masks : [M, 1, 1, 1, 1, C] or [M, 1, C, 1, 1, 1]
        if self.image_data_format == 'channels_first':
            x = x * self.masks[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
        elif self.image_data_format == 'channels_last':
            x = x * self.masks[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
        x = tf.concat(tf.split(x, self.n), axis=1)
        return tf.squeeze(x, axis=0)


def generate_masks_(m: int, n: int, s: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by n, m, s params.
    Results of this function are stochastic, that is, calls with the same sets
    of arguments might generate outputs of different shapes. Check generate_masks
    and generation_wrapper function for more deterministic behaviour.
    :param m: int, number of ones in each mask
    :param n: int, number of masks in the set
    :param s: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors
    """

    total_positions = int(m * s)
    masks = []

    for _ in range(n):
        new_vector = np.zeros([total_positions])
        idx = np.random.choice(range(total_positions), m, replace=False)
        new_vector[idx] = 1
        masks.append(new_vector)

    masks = np.array(masks)
    # drop useless positions
    masks = masks[:, ~np.all(masks == 0, axis=0)]
    return masks


def generate_masks(m: int, n: int, s: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by n, m, s params.
    Resulting masks are required to have fixed features size as it's described in [1].
    Since process of masks generation is stochastic therefore function evaluates
    generate_masks_ multiple times till expected size is acquired.
    :param m: int, number of ones in each mask
    :param n: int, number of masks in the set
    :param s: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors
    References
    [1] `Masksembles for Uncertainty Estimation: Supplementary Material`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua
    """

    masks = generate_masks_(m, n, s)
    # hardcoded formula for expected size, check reference
    expected_size = int(m * s * (1 - (1 - 1 / s) ** n))
    while masks.shape[1] != expected_size:
        masks = generate_masks_(m, n, s)
    return masks


def generation_wrapper(c: int, n: int, scale: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by c, n, scale params.
     Allows to generate masks sets with predefined features number c. Particularly
     convenient to use in torch-like layers where one need to define shapes inputs
     tensors beforehand.
    :param c: int, number of channels in generated masks
    :param n: int, number of masks in the set
    :param scale: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors
    """

    if c < 8:
        raise ValueError("Masksembles approach couldn't be used in such setups where "
                         f"number of channels is less then 8. Current value is (channels={c}). "
                         "Please increase number of features in your layer or remove this "
                         "particular instance of Masksembles from your architecture.")

    if scale > 6.:
        raise ValueError("Masksembles approach couldn't be used in such setups where "
                         f"scale parameter is larger then 6. Current value is (scale={scale}).")

    # inverse formula for number of active features in masks
    active_features = int(int(c) / (scale * (1 - (1 - 1 / scale) ** n)))

    # FIXME this piece searches for scale parameter value that generates
    #  proper number of features in masks, sometimes search is not accurate
    #  enough and masks.shape != c. Could fix it with binary search.
    masks = generate_masks(active_features, n, scale)
    for s in np.linspace(max(0.8 * scale, 1.0), 1.5 * scale, 300):
        if masks.shape[-1] >= c:
            break
        masks = generate_masks(active_features, n, s)
    new_upper_scale = s

    if masks.shape[-1] != c:
        for s in np.linspace(max(0.8 * scale, 1.0), new_upper_scale, 1000):
            if masks.shape[-1] >= c:
                break
            masks = generate_masks(active_features, n, s)

    if masks.shape[-1] != c:
        raise ValueError("generation_wrapper function failed to generate masks with "
                         "requested number of features. Please try to change scale parameter")

    return masks