import logging

import keras

from nn.blocks import conv_block_d, sagan_block

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiscriminatorV1:
    def __init__(self, input_size=64, use_self_attn=True, norm="none"):
        logger.info(f"Initializing Discriminator with (input_size={input_size}, "
                    f"use_self_attn={use_self_attn}, norm='{norm}')")
        self.input_size = input_size
        self.use_self_attn = use_self_attn
        self.norm = norm
        self.conv2d = keras.layers.Conv2D(1, kernel_size=4, use_bias=False, padding="same")
        self.config = {}
        pass

    def build(self, shape):
        use_norm = False if (self.norm == 'none') else True

        inputs = keras.Input(shape)

        x = conv_block_d(inputs, 64, False)
        x = conv_block_d(x, 128, use_norm, norm=self.norm)
        x = conv_block_d(x, 256, use_norm, norm=self.norm)
        x = sagan_block(x, 256) if self.use_self_attn else x

        activ_map_size = self.input_size // 8
        while activ_map_size > 8:
            x = conv_block_d(x, 256, use_norm, norm=self.norm)
            x = sagan_block(x, 256) if self.use_self_attn else x
            activ_map_size = activ_map_size // 2
            pass

        # use_bias should be True
        out = self.conv2d(x)
        model = keras.Model(inputs=inputs, outputs=out)
        logger.info("Initialized Discriminator Model")
        self.config = model.get_config()
        return model

    # noinspection PyTypeChecker
    def get_config(self):
        conf = {
            'input_size': self.input_size,
            'normalization': self.norm
        }
        return dict(list(self.config.items()) + list(conf.items()))

    pass
