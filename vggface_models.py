"""
VGGFace models for Keras.

# Notes:
- Resnet50 and VGG16  are modified architectures from Keras Application folder. [Keras](https://keras.io)

- Squeeze and excitation block is taken from  [Squeeze and Excitation Networks in
 Keras](https://github.com/titu1994/keras-squeeze-excite-network) and modified.
"""

# Source: https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/models.py

import warnings

from keras import backend as K
from keras import layers
from keras_vggface import utils
from keras.engine.topology import get_source_inputs
from keras.layers import (
    Flatten,
    Dense,
    Input,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Activation,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    AveragePooling2D,
    Reshape, multiply
)
from keras.models import Model
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
# noinspection PyProtectedMember
from keras_applications.imagenet_utils import _obtain_input_shape


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
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        pass
    # Create model.
    model = Model(inputs, x, name='vggface_vgg16')  # load weights
    if weights == 'vggface':
        if include_top:
            weights_path = get_file('rcmalli_vggface_tf_vgg16.h5',
                                    utils.VGG16_WEIGHTS_PATH,
                                    cache_subdir=utils.VGGFACE_DIR)
        else:
            weights_path = get_file('rcmalli_vggface_tf_notop_vgg16.h5',
                                    utils.VGG16_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir=utils.VGGFACE_DIR)
        model.load_weights(weights_path, by_name=True)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
            pass

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
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), use_bias=bias, name=conv1_reduce_name)(
        input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, use_bias=bias,
               padding='same', name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=bias, name=conv1_increase_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def resnet_conv_block(input_tensor, kernel_size, filters, stage, block,
                      strides=(2, 2), bias=False):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv1_proj_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_proj"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), strides=strides, use_bias=bias,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=bias,
                      name=conv1_proj_name)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=conv1_proj_name + "/bn")(
        shortcut)

    x = layers.add([x, shortcut])
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
        64, (7, 7), use_bias=False, strides=(2, 2), padding='same',
        name='conv1/7x7_s2')(img_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = RESNET(x)

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
        inputs = get_source_inputs(input_tensor)
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
                                    utils.RESNET50_WEIGHTS_PATH,
                                    cache_subdir=utils.VGGFACE_DIR)
            pass
        else:
            weights_path = get_file('rcmalli_vggface_tf_notop_resnet50.h5',
                                    utils.RESNET50_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir=utils.VGGFACE_DIR)
            pass
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='classifier')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')
                pass
            pass

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

    m = layers.add([se, shortcut])
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

    m = layers.add([x, se])
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

    x = senet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = senet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = senet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = senet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = senet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = senet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = senet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = senet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = senet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = senet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = senet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

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
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vggface_senet50')

    # load weights
    if weights == 'vggface':
        if include_top:
            weights_path = get_file('rcmalli_vggface_tf_senet50.h5',
                                    utils.SENET50_WEIGHTS_PATH,
                                    cache_subdir=utils.VGGFACE_DIR)
        else:
            weights_path = get_file('rcmalli_vggface_tf_notop_senet50.h5',
                                    utils.SENET50_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir=utils.VGGFACE_DIR)
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='classifier')
                layer_utils.convert_dense_weights_data_format(dense, shape,
                                                              'channels_first')

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
            input_ = Conv2D(filters, (3, 3), activation='relu', padding='same', name=f"{name_conv}_{i + 1}")(input_)

        return MaxPooling2D((2, 2), strides=(2, 2), name=name_pool)(input_)

    # Block 1
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    x = input_tensor.copy()

    x = _conv_block(x, 64, 2, name_conv='conv1', name_pool='pool1')
    x = _conv_block(x, 128, 2, name_conv='conv2', name_pool='pool2')
    x = _conv_block(x, 256, 3, name_conv='conv3', name_pool='pool3')
    x = _conv_block(x, 512, 3, name_conv='conv4', name_pool='pool4')
    return x


# noinspection PyPep8Naming
def RESNET(x):
    def _resnet_block(input_, stage: int, blocks: int, filters: list[int]):
        strides = (1, 1) if stage == 2 else (2, 2)
        for block in range(1, blocks + 1):
            if block == 1:
                input_ = resnet_conv_block(input_, 3, filters, stage, block, strides=strides)
            else:
                input_ = resnet_identity_block(input_, 3, filters, stage, block)
                pass
            pass
        return input_

    x = _resnet_block(x, stage=2, blocks=3, filters=[64, 64, 256])
    x = _resnet_block(x, stage=3, blocks=4, filters=[128, 128, 512])
    x = _resnet_block(x, stage=4, blocks=6, filters=[256, 256, 1024])
    x = _resnet_block(x, stage=5, blocks=3, filters=[512, 512, 2048])

    # x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    # x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    # x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    # x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    # x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    # x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    # x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    # x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    # x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    # x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    # x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    # x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    # x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    # x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    # x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    # x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)
    return x
