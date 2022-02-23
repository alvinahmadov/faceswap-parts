import logging

import keras

from nn.blocks import upscale_nn, upscale_ps, sagan_block, res_block, conv_block

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecoderV1:
    def __init__(self, model_capacity="standard",
                 input_size=8, output_size=64,
                 use_self_attn=True, norm='none'):
        logger.info(f"Initializing Decoder with (model_capacity='{model_capacity}', "
                    f"input_size={input_size}, output_size={output_size}, "
                    f"use_self_attn={use_self_attn}, norm='{norm}')")
        self.coef = 2 if model_capacity == "lite" else 1
        self.upscale_block = upscale_nn if model_capacity == "lite" else upscale_ps
        self.input_size = input_size
        self.norm = norm
        self.use_self_attn = use_self_attn
        self.output_size = output_size

        self.alpha = keras.layers.Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")
        self.bgr = keras.layers.Conv2D(3, kernel_size=5, padding='same', activation="tanh")
        self.conv2d = keras.layers.Conv2D(3, kernel_size=5, padding='same', activation="tanh")
        self.config = {}

    def build(self, shape) -> keras.Model:
        use_norm = False if (self.norm == 'none') else True

        inputs = keras.Input(shape)

        x = self.upscale_block(inputs, 256 // self.coef, use_norm, norm=self.norm)
        x = self.upscale_block(x, 128 // self.coef, use_norm, norm=self.norm)
        x = sagan_block(x, 128 // self.coef) if self.use_self_attn else x
        x = self.upscale_block(x, 64 // self.coef, use_norm, norm=self.norm)
        x = res_block(x, 64 // self.coef, norm=self.norm)
        x = sagan_block(x, 64 // self.coef) \
            if self.use_self_attn else conv_block(x, 64 // self.coef, strides=1)

        outputs = []
        activ_map_size = self.input_size * 8

        while activ_map_size < self.output_size:
            outputs.append(self.conv2d(x))
            x = self.upscale_block(x, 64 // self.coef, use_norm, norm=self.norm)
            x = conv_block(x, 64 // self.coef, strides=1)
            activ_map_size *= 2
            pass

        alpha = self.alpha(x)
        bgr = self.bgr(x)
        out = keras.layers.concatenate([alpha, bgr])
        outputs.append(out)
        model = keras.Model(inputs=inputs, outputs=outputs)
        logger.info("Initialized Decoder Model")
        self.config = model.get_config()
        return model

    # noinspection PyTypeChecker
    def get_config(self) -> dict:
        conf = {
            'output_size': self.output_size,
            'model_capacity': "lite" if self.coef == 2 else "standard",
            'input_size': self.input_size,
            'coefficient': self.coef,
            'normalization': self.norm
        }
        return dict(list(self.config.items()) + list(conf.items()))

    pass
