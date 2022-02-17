import keras.backend as K
import tensorflow as tf
from keras import regularizers
from keras.layers import (
    Conv2D, Reshape, Lambda, Permute, LeakyReLU,
    Softmax, Activation, UpSampling2D, BatchNormalization,
    add, concatenate, multiply
)

from .custom_inits.icnr_initializer import icnr_keras
from .group_normalization import GroupNormalization
from .instance_normalization import InstanceNormalization
from .layers.scale_layer import Scale
from .pixel_shuffler import PixelShuffler

KERN_INIT = 'he_normal'
W_L2 = 1e-4


def sagan_block(inputs, num_input_channels, squeeze_factor=8):
    """
    SAGAN block

    References
    ------
    Self-Attention Generative Adversarial Network (SAGAN) (https://arxiv.org/pdf/1805.08318.pdf)

    Code borrows from https://github.com/taki0112/Self-Attention-GAN-Tensorflow
    """

    assert num_input_channels // squeeze_factor > 0, \
        f"Input channels must be >= {squeeze_factor}, got num_input_channels={num_input_channels}"
    shape = inputs.get_shape().as_list()

    f = Conv2D(num_input_channels // squeeze_factor, 1, kernel_regularizer=regularizers.l2(W_L2))(inputs)
    g = Conv2D(num_input_channels // squeeze_factor, 1, kernel_regularizer=regularizers.l2(W_L2))(inputs)
    h = Conv2D(num_input_channels, 1, kernel_regularizer=regularizers.l2(W_L2))(inputs)

    shape_f = f.get_shape().as_list()
    shape_g = g.get_shape().as_list()
    shape_h = h.get_shape().as_list()
    flat_f = Reshape((-1, shape_f[-1]))(f)
    flat_g = Reshape((-1, shape_g[-1]))(g)
    flat_h = Reshape((-1, shape_h[-1]))(h)

    s = Lambda(lambda x_: K.batch_dot(x_[0], Permute((2, 1))(x_[1])))([flat_g, flat_f])

    beta = Softmax(axis=-1)(s)
    o = Lambda(lambda x_: K.batch_dot(x_[0], x_[1]))([beta, flat_h])
    o = Reshape(shape[1:])(o)
    o = Scale()(o)

    return add([o, inputs])


def dan_block(inputs, num_input_channels, squeeze_factor=8, norm='none'):
    """
    Dual Attention Network block

    References
    ------
    Dual Attention Network for Scene Segmentation (https://arxiv.org/pdf/1809.02983.pdf)

    Code from https://github.com/junfu1115/DANet
    """
    assert num_input_channels // squeeze_factor > 0, \
        f"Input channels must be >= {squeeze_factor}, got num_input_channels={num_input_channels}"
    x = inputs
    shape = inputs.get_shape().as_list()

    # position attention module
    x_pam = Conv2D(num_input_channels, kernel_size=3, use_bias=False, padding="same",
                   kernel_regularizer=regularizers.l2(W_L2), kernel_initializer=KERN_INIT)(inputs)
    x_pam = Activation("relu")(x_pam)
    x_pam = normalization(x_pam, norm, num_input_channels)
    f_pam = Conv2D(num_input_channels // squeeze_factor, 1, kernel_regularizer=regularizers.l2(W_L2))(x_pam)
    g_pam = Conv2D(num_input_channels // squeeze_factor, 1, kernel_regularizer=regularizers.l2(W_L2))(x_pam)
    h_pam = Conv2D(num_input_channels, 1, kernel_regularizer=regularizers.l2(W_L2))(x_pam)

    shape_f_pam = f_pam.get_shape().as_list()
    shape_g_pam = g_pam.get_shape().as_list()
    shape_h_pam = h_pam.get_shape().as_list()

    flat_f_pam = Reshape((-1, shape_f_pam[-1]))(f_pam)
    flat_g_pam = Reshape((-1, shape_g_pam[-1]))(g_pam)
    flat_h_pam = Reshape((-1, shape_h_pam[-1]))(h_pam)

    s_pam = Lambda(lambda x_: K.batch_dot(x_[0], Permute((2, 1))(x_[1])))([flat_g_pam, flat_f_pam])
    beta_pam = Softmax(axis=-1)(s_pam)
    o_pam = Lambda(lambda x_: K.batch_dot(x_[0], x_[1]))([beta_pam, flat_h_pam])
    o_pam = Reshape(shape[1:])(o_pam)
    o_pam = Scale()(o_pam)
    out_pam = add([o_pam, x_pam])
    out_pam = Conv2D(num_input_channels, kernel_size=3, kernel_regularizer=regularizers.l2(W_L2),
                     kernel_initializer=KERN_INIT, use_bias=False, padding="same")(out_pam)
    out_pam = Activation("relu")(out_pam)
    out_pam = normalization(out_pam, norm, num_input_channels)

    # channel attention module
    x_chn = Conv2D(num_input_channels, kernel_size=3, kernel_regularizer=regularizers.l2(W_L2),
                   kernel_initializer=KERN_INIT, use_bias=False, padding="same")(x)
    x_chn = Activation("relu")(x_chn)
    x_chn = normalization(x_chn, norm, num_input_channels)
    shape_x_chn = x_chn.get_shape().as_list()
    flat_f_chn = Reshape((-1, shape_x_chn[-1]))(x_chn)
    flat_g_chn = Reshape((-1, shape_x_chn[-1]))(x_chn)
    flat_h_chn = Reshape((-1, shape_x_chn[-1]))(x_chn)
    s_chn = Lambda(lambda x_: K.batch_dot(Permute((2, 1))(x_[0]), x_[1]))([flat_g_chn, flat_f_chn])
    s_new_chn = Lambda(lambda x_: K.repeat_elements(K.max(x_, -1, keepdims=True), num_input_channels, -1))(s_chn)
    s_new_chn = Lambda(lambda x_: x_[0] - x_[1])([s_new_chn, s_chn])
    beta_chn = Softmax(axis=-1)(s_new_chn)
    o_chn = Lambda(lambda x_: K.batch_dot(x_[0], Permute((2, 1))(x_[1])))([flat_h_chn, beta_chn])
    o_chn = Reshape(shape[1:])(o_chn)
    o_chn = Scale()(o_chn)
    out_chn = add([o_chn, x_chn])
    out_chn = Conv2D(num_input_channels, kernel_size=3, kernel_regularizer=regularizers.l2(W_L2),
                     kernel_initializer=KERN_INIT, use_bias=False, padding="same")(out_chn)
    out_chn = Activation("relu")(out_chn)
    out_chn = normalization(out_chn, norm, num_input_channels)

    return add([out_pam, out_chn])


def normalization(inp, norm='none', group=16):
    x = inp
    if norm == 'layernorm':
        x = GroupNormalization(group=group)(x)
    elif norm == 'batchnorm':
        x = BatchNormalization()(x)
    elif norm == 'groupnorm':
        x = GroupNormalization(group=16)(x)
    elif norm == 'instancenorm':
        x = InstanceNormalization()(x)
    elif norm == 'hybrid':
        if group % 2 == 1:
            raise ValueError(f"Output channels must be an even number for hybrid norm, received {group}.")
        f = group
        x0 = Lambda(lambda arg: arg[..., :f // 2])(x)
        x1 = Lambda(lambda arg: arg[..., f // 2:])(x)
        x0 = Conv2D(f // 2, kernel_size=1, kernel_regularizer=regularizers.l2(W_L2),
                    kernel_initializer=KERN_INIT)(x0)
        x1 = InstanceNormalization()(x1)
        x = concatenate([x0, x1], axis=-1)
    else:
        x = x
    return x


def _conv_block(input_tensor, group, use_norm=False, kernel_size=3, strides=2, w_l2=W_L2, norm='none',
                activation: str = 'relu'):
    x = Conv2D(group, kernel_size=kernel_size, strides=strides, kernel_regularizer=regularizers.l2(w_l2),
               kernel_initializer=KERN_INIT, use_bias=False, padding="same")(input_tensor)
    x = Activation("relu")(x) if activation == 'relu' else LeakyReLU(alpha=0.2)(x)
    x = normalization(x, norm, group) if use_norm else x
    return x


def conv_block(input_tensor, group, use_norm=False, strides=2, w_l2=W_L2, norm='none'):
    return _conv_block(input_tensor, group, use_norm, strides=strides, w_l2=w_l2, norm=norm)


def conv_block_d(input_tensor, group, use_norm=False, w_l2=W_L2, norm='none'):
    return _conv_block(input_tensor, group, use_norm, 4, 2, w_l2, norm,
                       activation='leakyrelu')


def res_block(input_tensor, group, use_norm=False, w_l2=W_L2, norm='none'):
    x = Conv2D(group, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2),
               kernel_initializer=KERN_INIT, use_bias=False, padding="same")(input_tensor)
    x = LeakyReLU(alpha=0.2)(x)
    x = normalization(x, norm, group) if use_norm else x
    x = Conv2D(group, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2),
               kernel_initializer=KERN_INIT, use_bias=False, padding="same")(x)
    x = add([x, input_tensor])
    x = LeakyReLU(alpha=0.2)(x)
    x = normalization(x, norm, group) if use_norm else x
    return x


def spade_res_block(input_tensor, cond_input_tensor, group, use_norm=True, norm='none'):
    """
    Semantic Image Synthesis with Spatially-Adaptive Normalization
    Taesung Park, Ming-Yu Liu, Ting-Chun Wang, Jun-Yan Zhu
    https://arxiv.org/abs/1903.07291

    Note:
        SPADE just works like a charm. 
        It speeds up training alot and is also a very promosing approach for solving profile face generation issue.
        *(This implementation can be wrong since I haven't finished reading the paper. 
          The author hasn't release their code either (https://github.com/NVlabs/SPADE).)
    """

    def spade(input_tensor_, cond_input_tensor_, group_, use_norm_=True, norm_='none'):
        x_ = normalization(input_tensor_, norm_, group_) if use_norm_ else input_tensor_
        y_ = Conv2D(128, kernel_size=3, kernel_regularizer=regularizers.l2(W_L2),
                    kernel_initializer=KERN_INIT, padding='same')(cond_input_tensor_)
        y_ = Activation('relu')(y_)
        gamma = Conv2D(group_, kernel_size=3, kernel_regularizer=regularizers.l2(W_L2),
                       kernel_initializer=KERN_INIT, padding='same')(y_)
        beta = Conv2D(group_, kernel_size=3, kernel_regularizer=regularizers.l2(W_L2),
                      kernel_initializer=KERN_INIT, padding='same')(y_)
        x_ = add([x_, multiply([x_, gamma])])
        x_ = add([x_, beta])
        return x_

    x = spade(input_tensor, cond_input_tensor, group, use_norm, norm)
    x = Activation('relu')(x)
    x = reflect_padding_2d(x)
    x = Conv2D(group, kernel_size=3, kernel_regularizer=regularizers.l2(W_L2),
               kernel_initializer=KERN_INIT, use_bias=not use_norm)(x)
    x = spade(x, cond_input_tensor, group, use_norm, norm)
    x = Activation('relu')(x)
    x = reflect_padding_2d(x)
    x = Conv2D(group, kernel_size=3, kernel_regularizer=regularizers.l2(W_L2),
               kernel_initializer=KERN_INIT)(x)
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def upscale_ps(input_tensor, group, use_norm=False, w_l2=W_L2, norm='none'):
    x = input_tensor
    x = Conv2D(group * 4, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2),
               kernel_initializer=icnr_keras, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = normalization(x, norm, group) if use_norm else x
    x = PixelShuffler()(x)
    return x


def reflect_padding_2d(x, pad=1):
    return Lambda(lambda x_: tf.pad(x_, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT'))(x)


def upscale_nn(input_tensor, group, use_norm=False, w_l2=W_L2, norm='none'):
    x = UpSampling2D()(input_tensor)
    x = reflect_padding_2d(x, 1)
    x = Conv2D(group, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), kernel_initializer=KERN_INIT)(x)
    x = normalization(x, norm, group) if use_norm else x
    return x
