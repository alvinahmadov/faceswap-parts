import keras
import logging

from nn.blocks import upscale_nn, upscale_ps, conv_block, sagan_block

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EncoderV1:
    def __init__(self, model_capacity="standard", input_size=64, use_self_attn=True, norm='none'):
        logger.info(f"Initializing Encoder with (model_capacity='{model_capacity}', "
                    f"input_size={input_size}, use_self_attn={use_self_attn}, norm='{norm}')")
        self.model = None
        self.coef = 2 if model_capacity == "lite" else 1
        self.input_size = input_size
        self.upscale_block = upscale_nn if model_capacity == "lite" else upscale_ps
        self.norm: str = norm
        self.use_self_attn = use_self_attn

        self.conv2d = keras.layers.Conv2D(64 // self.coef, kernel_size=5, use_bias=False, padding="same")
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(2048 if (model_capacity == "lite" and self.input_size > 64) else 1024)
        self.dense2 = keras.layers.Dense(4 * 4 * 1024 // (self.coef ** 2))
        self.reshape = keras.layers.Reshape((4, 4, 1024 // (self.coef ** 2)))
        self.config = {}
        pass

    def build(self, shape) -> keras.Model:
        use_norm = False if (self.norm == 'none') else True

        inputs = keras.Input(shape)

        # use_bias should be True
        output = self.conv2d(inputs)
        output = conv_block(output, 128 // self.coef)
        output = conv_block(output, 256 // self.coef, use_norm, norm=self.norm)
        output = sagan_block(output, 256 // self.coef) if self.use_self_attn else output
        output = conv_block(output, 512 // self.coef, use_norm, norm=self.norm)
        output = sagan_block(output, 512 // self.coef) if self.use_self_attn else output
        output = conv_block(output, 1024 // (self.coef ** 2), use_norm, norm=self.norm)

        activ_map_size = self.input_size // 16
        while activ_map_size > 4:
            output = conv_block(output, 1024 // (self.coef ** 2), use_norm, norm=self.norm)
            activ_map_size = activ_map_size // 2
            pass

        output = self.dense1(self.flatten(output))
        output = self.dense2(output)
        output = self.reshape(output)
        output = self.upscale_block(output, 512 // self.coef, use_norm, norm=self.norm)
        model = keras.Model(inputs=inputs, outputs=output)
        logger.info("Initialized Encoder Model")
        self.config = model.get_config()
        return model

    # noinspection PyTypeChecker
    def get_config(self):
        conf = {
            'model_capacity': "lite" if self.coef == 2 else "standard",
            'input_size': self.input_size,
            'coefficient': self.coef,
            'normalization': self.norm
        }
        return dict(list(self.config.items()) + list(conf.items()))

    pass
