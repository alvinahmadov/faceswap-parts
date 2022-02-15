"""
VGGFace models for Keras.

# Notes:
- Resnet50 and VGG16  are modified architectures from Keras Application folder. [Keras](https://keras.io)

- Squeeze and excitation block is taken from  [Squeeze and Excitation Networks in
 Keras](https://github.com/titu1994/keras-squeeze-excite-network) and modified.
"""

# Source: https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/models.py

import warnings

from keras.layers import BatchNormalization
from tensorflow.python.keras import backend as K

from tensorflow.python.keras.layers import (
    Input, Activation, Conv2D, Reshape,
    Flatten, Dense, MaxPooling2D, AveragePooling2D,
    GlobalAveragePooling2D, GlobalMaxPooling2D,
    add, multiply
)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils.data_utils import get_file
# noinspection PyProtectedMember
from keras_applications.imagenet_utils import _obtain_input_shape

V1_LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v1.npy'
V2_LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v2.npy'

VGG16_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5'
VGG16_WEIGHTS_PATH_NO_TOP = \
    'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_vgg16.h5'

RESNET50_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_resnet50.h5'
RESNET50_WEIGHTS_PATH_NO_TOP = \
    'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_resnet50.h5'

SENET50_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_senet50.h5'
SENET50_WEIGHTS_PATH_NO_TOP = \
    'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_senet50.h5'

VGGFACE_DIR = 'models/vggface'


def vgg16(include_top=True, weights='vggface',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=2622):
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = conv_block(img_input)
    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, name='fc6')(x)
        x = Activation('relu', name='fc6/relu')(x)
        x = Dense(4096, name='fc7')(x)
        x = Activation('relu', name='fc7/relu')(x)
        x = Dense(classes, name='fc8')(x)
        x = Activation('softmax', name='fc8/softmax')(x)
        pass
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
            pass
        pass
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
        pass
    # Create model.
    model = Model(inputs, x, name='vggface_vgg16')  # load weights
    if weights == 'vggface':
        if include_top:
            weights_path = get_file('rcmalli_vggface_tf_vgg16.h5',
                                    VGG16_WEIGHTS_PATH,
                                    cache_subdir=VGGFACE_DIR)
        else:
            weights_path = get_file('rcmalli_vggface_tf_notop_vgg16.h5',
                                    VGG16_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir=VGGFACE_DIR)
        model.load_weights(weights_path, by_name=True)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='pool5')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc6')
                layer_utils.convert_dense_weights_data_format(
                    dense, shape, 'channels_first'
                )
                pass

            if K.backend() == 'tensorflow':
                warnings.warn(
                    'You are using the TensorFlow backend, yet you '
                    'are using the Theano '
                    'image data format convention '
                    '(`image_data_format="channels_first"`). '
                    'For best performance, set '
                    '`image_data_format="channels_last"` in '
                    'your Keras config '
                    'at ~/.keras/keras.json.'
                )
                pass
            pass
        pass
    return model


def resnet_identity_block(input_tensor, kernel_size, filters, stage, block,
                          bias=False):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = f"conv{stage}_{block}_1x1_reduce"
    conv1_increase_name = f"conv{stage}_{block}_1x1_increase"
    conv3_name = f"conv{stage}_{block}_3x3"

    x = Conv2D(filters1, (1, 1), use_bias=bias, name=conv1_reduce_name)(
        input_tensor)
    x = BatchNormalization(axis=bn_axis, name=f"{conv1_reduce_name}/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, use_bias=bias, padding='same', name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=f"{conv3_name}/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=bias, name=conv1_increase_name)(x)
    x = BatchNormalization(axis=bn_axis, name=f"{conv1_increase_name}/bn")(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def resnet_conv_block(input_tensor, kernel_size, filters, stage, block,
                      strides=(2, 2), bias=False):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = f"conv{stage}_{block}_1x1_reduce"
    conv1_increase_name = f"conv{stage}_{block}_1x1_increase"
    conv1_proj_name = f"conv{stage}_{block}_1x1_proj"
    conv3_name = f"conv{stage}_{block}_3x3"

    x = Conv2D(filters1, (1, 1), strides=strides, use_bias=bias,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=f"{conv1_reduce_name}/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=f"{conv3_name}/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=f"{conv1_increase_name}/bn")(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=bias, name=conv1_proj_name)(
        input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=f"{conv1_proj_name}/bn")(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


# noinspection PyPep8Naming
def RESNET50(include_top=True, weights='vggface',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=8631):
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
            pass
        pass
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
        pass

    x = Conv2D(
        64, (7, 7), use_bias=False, strides=(2, 2), padding='same', name='conv1/7x7_s2'
    )(img_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = _resnet_blocks(x)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='classifier')(x)
        pass
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
            pass
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
            pass
        pass

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
        pass
    else:
        inputs = img_input
        pass
    # Create model.
    model = Model(inputs, x, name='vggface_resnet50')

    # load weights
    if weights == 'vggface':
        if include_top:
            weights_path = get_file('rcmalli_vggface_tf_resnet50.h5',
                                    RESNET50_WEIGHTS_PATH,
                                    cache_subdir=VGGFACE_DIR)
            pass
        else:
            weights_path = get_file('rcmalli_vggface_tf_notop_resnet50.h5',
                                    RESNET50_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir=VGGFACE_DIR)
            pass
        model.load_weights(weights_path)

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
            pass
        pass
    elif weights is not None:
        model.load_weights(weights)
        pass

    return model


def senet_se_block(input_tensor, stage, block, compress_rate=16, bias=False):
    conv1_down_name = f"conv{stage}_{block}_1x1_down"
    conv1_up_name = f"conv{stage}_{block}_1x1_up"

    num_channels = int(input_tensor.shape[-1])
    bottle_neck = int(num_channels // compress_rate)

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, num_channels))(se)
    se = Conv2D(bottle_neck, (1, 1), use_bias=bias, name=conv1_down_name)(se)
    se = Activation('relu')(se)
    se = Conv2D(num_channels, (1, 1), use_bias=bias, name=conv1_up_name)(se)
    se = Activation('sigmoid')(se)

    x = input_tensor
    x = multiply([x, se])
    return x


def senet_conv_block(input_tensor, kernel_size, filters,
                     stage, block, bias=False, strides=(2, 2)):
    """
    Squeeze-and-Excitation Networks implementation

    References
    -------

    https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf

    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
        pass
    else:
        bn_axis = 1
        pass

    conv1_reduce_name = f"conv{stage}_{block}_1x1_reduce"
    conv1_increase_name = f"conv{stage}_{block}_1x1_increase"
    conv1_proj_name = f"conv{stage}_{block}_1x1_proj"
    conv3_name = f"conv{stage}_{block}_3x3"

    x = Conv2D(filters1, (1, 1), use_bias=bias, strides=strides, name=conv1_reduce_name)(
        input_tensor
    )
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias, name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    se = senet_se_block(x, stage=stage, block=block, bias=True)

    shortcut = Conv2D(filters3, (1, 1), use_bias=bias, strides=strides, name=conv1_proj_name)(
        input_tensor
    )
    shortcut = BatchNormalization(axis=bn_axis, name=conv1_proj_name + "/bn")(shortcut)

    m = add([se, shortcut])
    m = Activation('relu')(m)
    return m


def senet_identity_block(input_tensor, kernel_size,
                         filters, stage, block, bias=False):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), use_bias=bias,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    se = senet_se_block(x, stage=stage, block=block, bias=True)

    m = add([x, se])
    m = Activation('relu')(m)

    return m


# noinspection PyPep8Naming
def SENET50(include_top=True, weights='vggface',
            input_tensor=None, input_shape=None,
            pooling=None,
            classes=8631):
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
        pass
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
        pass
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
        pass

    x = Conv2D(
        64, (7, 7), use_bias=False, strides=(2, 2), padding='same',
        name='conv1/7x7_s2')(img_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = _senet_blocks(x)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='classifier')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vggface_senet50')

    # load weights
    if weights == 'vggface':
        if include_top:
            weights_path = get_file('rcmalli_vggface_tf_senet50.h5',
                                    SENET50_WEIGHTS_PATH,
                                    cache_subdir=VGGFACE_DIR)
        else:
            weights_path = get_file('rcmalli_vggface_tf_notop_senet50.h5',
                                    SENET50_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir=VGGFACE_DIR)
        model.load_weights(weights_path)

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    elif weights is not None:
        model.load_weights(weights)

    return model


def conv_block(input_tensor):
    def _conv_block(input_, filters, depth, name_conv, name_pool):
        for i in range(depth):
            input_ = Conv2D(
                filters, (3, 3), activation='relu', padding='same', name=f"{name_conv}_{i + 1}"
            )(input_)

        return MaxPooling2D((2, 2), strides=(2, 2), name=name_pool)(input_)

    x = input_tensor

    x = _conv_block(x, 64, 2, name_conv='conv1', name_pool='pool1')
    x = _conv_block(x, 128, 2, name_conv='conv2', name_pool='pool2')
    x = _conv_block(x, 256, 3, name_conv='conv3', name_pool='pool3')
    x = _conv_block(x, 512, 3, name_conv='conv4', name_pool='pool4')
    return x


def _net_blocks(input_, stage: int, blocks: int, filters: list, net: str = 'RES'):
    strides = (1, 1) if stage == 2 else (2, 2)
    if net == 'RES':
        conv_block_fn = resnet_conv_block
        identity_block_fn = resnet_identity_block
    elif net == 'SE':
        conv_block_fn = senet_conv_block
        identity_block_fn = senet_identity_block
    else:
        raise ValueError("Net must be RESNET or SENET")

    for block in range(1, blocks + 1):
        if block == 1:
            input_ = conv_block_fn(input_, 3, filters, stage, block, strides=strides)
        else:
            input_ = identity_block_fn(input_, 3, filters, stage, block)
            pass
        pass
    return input_


def _resnet_blocks(x):
    x = _net_blocks(x, stage=2, blocks=3, filters=[64, 64, 256])
    x = _net_blocks(x, stage=3, blocks=4, filters=[128, 128, 512])
    x = _net_blocks(x, stage=4, blocks=6, filters=[256, 256, 1024])
    x = _net_blocks(x, stage=5, blocks=3, filters=[512, 512, 2048])

    return x


def _senet_blocks(x):
    net = "SE"
    x = _net_blocks(x, stage=2, blocks=3, filters=[64, 64, 256], net=net)
    x = _net_blocks(x, stage=3, blocks=4, filters=[128, 128, 512], net=net)
    x = _net_blocks(x, stage=4, blocks=6, filters=[256, 256, 1024], net=net)
    x = _net_blocks(x, stage=5, blocks=3, filters=[512, 512, 2048], net=net)

    return x
