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
""" Custom loss functions. """

import tensorflow.keras.backend as K
from tensorflow.keras.layers import ZeroPadding3D
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model


class VGG_LOSS(object):
    '''
    for 2D
    '''
    def __init__(self, image_shape):
        self.image_shape = image_shape

        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        self.model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        self.model.trainable = False

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
        return K.mean(K.square(self.model(y_true) - self.model(y_pred)))

class MSE_GD_LOSS(object):
    def __init__(self, p=1.25, weight=1):
        super().__init__()
        self.p = p  # power of tv loss
        self.weight = weight  # weight of tv loss

    def mse_gd_loss(self, y_true, y_pred):
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        gd = self.gradient_difference_loss(y_true, y_pred, self.p)
        loss = mse+self.weight * gd
        return loss

    def gradient_difference_loss(self, y_true, y_pred, p=1.25):
        '''
        formula: https://openreview.net/pdf?id=rJevSbniM
        :param x: 5D tensor, (batch, channel, x, y, z)
        :return:
        '''
        dx_true = K.abs( y_true[:, :, 1:, :, :] - y_true[:, :, :-1, :, :] )
        dy_true = K.abs( y_true[:, :, :, 1:, :] - y_true[:, :, :, :-1, :] )
        dz_true = K.abs( y_true[:, :, :, :, 1:] - y_true[:, :, :, :, :-1] )
        dx_pred = K.abs( y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :] )
        dy_pred = K.abs( y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :] )
        dz_pred = K.abs( y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1] )
        total = tf.math.reduce_sum(K.pow(dx_true, p) - K.pow(dx_pred, p)) \
                + tf.math.reduce_sum(K.pow(dy_true, p) - K.pow(dy_pred, p)) \
                + tf.math.reduce_sum(K.pow(dz_true, p) - K.pow(dz_pred, p))
        return total

class L2TV(object):
    def __init__(self, p=1.25, weight=1):
        # super().__init__(p, weight)
        super().__init__()
        self.p = p # power of tv
        self.weight = weight # regularisation parameter of tv term

    def l2_tv(self, y_true, y_pred):
        """Computes l2 loss + tv penalty.
        ```
        l2 = tf.keras.losses.MeanSquaredError()
        loss = l2(y_true, y_pred)
        ```
        where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss
        Args:
          y_true: tensor of true targets.
          y_pred: tensor of predicted targets.
          delta: A float, the point where the Huber loss function changes from a
            quadratic to linear.
        Returns:
          Tensor
        """
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        tv = self.total_variation_loss(y_pred, self.p)
        loss = mse + self.weight * tv
        return loss

    def total_variation_loss(self, x, p=1.25):
        '''
        formula: https://raghakot.github.io/keras-vis/vis.regularizers/#totalvariation
        homogeneity: https://link.springer.com/content/pdf/10.1023/B:JMIV.0000011325.36760.1e.pdf
        :param x: 5D tensor, (batch, channel, x, y, z)
        :return:
        '''
        x = ZeroPadding3D(padding=1)(x)
        a = K.square( x[:, :, :-2, 1:-1, 1:-1] - x[:, :, 2:, 1:-1, 1:-1] )
        b = K.square( x[:, :, 1:-1, :-2, 1:-1] - x[:, :, 1:-1, 2:, 1:-1] )
        c = K.square( x[:, :, 1:-1, 1:-1, :-2] - x[:, :, 1:-1, 1:-1, 2:] )
        total = tf.math.reduce_sum(K.pow(a, p/2)) \
                + tf.math.reduce_sum(K.pow(b, p/2)) \
                + tf.math.reduce_sum(K.pow(c, p/2))
        return total

class L2L2(object):
    def __init__(self, weight=1):
        # super().__init__(p, weight)
        super().__init__()
        self.weight = weight # regularisation parameter of l2 term

    def l2_l2(self, y_true, y_pred):
        """Computes l2 loss + tv penalty.
        ```
        l2 = tf.keras.losses.MeanSquaredError()
        loss = l2(y_true, y_pred)
        ```
        where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss
        Args:
          y_true: tensor of true targets.
          y_pred: tensor of predicted targets.
          delta: A float, the point where the Huber loss function changes from a
            quadratic to linear.
        Returns:
          Tensor
        """
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        l2 = self.l2_norm(y_pred)
        loss = mse + self.weight * l2
        return loss

    def l2_norm(self, x):
        '''
        formula: https://raghakot.github.io/keras-vis/vis.regularizers/#totalvariation
        homogeneity: https://link.springer.com/content/pdf/10.1023/B:JMIV.0000011325.36760.1e.pdf
        :param x: 5D tensor, (batch, channel, x, y, z)
        :return:
        '''
        total = tf.math.reduce_sum(K.square(x))
        return total