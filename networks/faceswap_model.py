import os

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from networks.losses import (
    first_order, cyclic_loss, adversarial_loss,
    reconstruction_loss, edge_loss, perceptual_loss
)
from networks.nn_blocks import *


class FaceswapModel:
    """
    Faceswap model
    
    Attributes
    ------
    num_gen_input_channels : int
     Number of generator input channels

    num_disc_input_channels : int
     Number of discriminator input channels

    learning_rate_gen : float
     Learning rate of the generator

    learning_rate_disc : float
     Learning rate of the discriminator
    """

    def __init__(self, **arch_config):
        """
        Parameters
        ----------
        arch_config : dict
         A dictionary that contains architecture configurations.
        """
        self.num_gen_input_channels = 3
        self.num_disc_input_channels = 6
        self.image_shape = arch_config['IMAGE_SHAPE']
        self.learning_rate_disc = 2e-4
        self.learning_rate_gen = 1e-4
        self.use_self_attn = arch_config['use_self_attn']
        self.norm = arch_config['norm']
        self.model_capacity = arch_config['model_capacity']
        self.enc_nc_out = 256 if self.model_capacity == "lite" else 512

        self.net_disc_train_src = None
        self.net_disc_train_dst = None

        self.net_gen_train_src = None
        self.net_gen_train_dst = None

        self.vggface_feats = None

        # define networks
        self.encoder = self.build_encoder(
            num_input_channels=self.num_gen_input_channels,
            input_size=self.image_shape[0],
            use_self_attn=self.use_self_attn,
            norm=self.norm,
            model_capacity=self.model_capacity
        )
        self.decoder_src = self.build_decoder(
            nc_in=self.enc_nc_out,
            input_size=8,
            output_size=self.image_shape[0],
            use_self_attn=self.use_self_attn,
            norm=self.norm,
            model_capacity=self.model_capacity
        )
        self.decoder_dst = self.build_decoder(
            nc_in=self.enc_nc_out,
            input_size=8,
            output_size=self.image_shape[0],
            use_self_attn=self.use_self_attn,
            norm=self.norm,
            model_capacity=self.model_capacity
        )
        self.net_disc_src = self.build_discriminator(
            nc_in=self.num_disc_input_channels,
            input_size=self.image_shape[0],
            use_self_attn=self.use_self_attn,
            norm=self.norm
        )
        self.net_disc_dst: Model = self.build_discriminator(
            nc_in=self.num_disc_input_channels,
            input_size=self.image_shape[0],
            use_self_attn=self.use_self_attn,
            norm=self.norm
        )
        x = Input(shape=self.image_shape)  # dummy input tensor
        self.net_gen_src: Model = Model(x, self.decoder_src(self.encoder(x)))
        self.net_gen_dst: Model = Model(x, self.decoder_dst(self.encoder(x)))

        # define variables
        (
            self.distorted_src, self.fake_src, self.mask_src, self.path_src,
            self.path_mask_src, self.path_abgr_src, self.path_bgr_src
        ) = self.define_variables(net_gen=self.net_gen_src)
        (
            self.distorted_dst, self.fake_dst, self.mask_dst, self.path_dst,
            self.path_mask_dst, self.path_abgr_dst, self.path_bgr_dst
        ) = self.define_variables(net_gen=self.net_gen_dst)
        self.real_src = Input(shape=self.image_shape)
        self.real_dst = Input(shape=self.image_shape)
        self.mask_eyes_src = Input(shape=self.image_shape)
        self.mask_eyes_dst = Input(shape=self.image_shape)
        pass

    @staticmethod
    def build_encoder(
            num_input_channels=3,
            input_size=64,
            use_self_attn=True,
            norm='none',
            model_capacity='standard'
    ):
        coef = 2 if model_capacity == "lite" else 1
        latent_dim = 2048 if (model_capacity == "lite" and input_size > 64) else 1024
        upscale_block = upscale_nn if model_capacity == "lite" else upscale_ps
        activ_map_size = input_size
        use_norm = False if (norm == 'none') else True

        inputs = Input(shape=(input_size, input_size, num_input_channels))

        # use_bias should be True
        x = Conv2D(64 // coef, kernel_size=5, use_bias=False, padding="same")(inputs)
        x = conv_block(x, 128 // coef)
        x = conv_block(x, 256 // coef, use_norm, norm=norm)
        x = self_attn_block(x, 256 // coef) if use_self_attn else x
        x = conv_block(x, 512 // coef, use_norm, norm=norm)
        x = self_attn_block(x, 512 // coef) if use_self_attn else x
        x = conv_block(x, 1024 // (coef ** 2), use_norm, norm=norm)

        activ_map_size = activ_map_size // 16
        while activ_map_size > 4:
            x = conv_block(x, 1024 // (coef ** 2), use_norm, norm=norm)
            activ_map_size = activ_map_size // 2
            pass

        x = Dense(latent_dim)(Flatten()(x))
        x = Dense(4 * 4 * 1024 // (coef ** 2))(x)
        x = Reshape((4, 4, 1024 // (coef ** 2)))(x)
        out = upscale_block(x, 512 // coef, use_norm, norm=norm)
        return Model(inputs=inputs, outputs=out)

    @staticmethod
    def build_decoder(nc_in=512,
                      input_size=8,
                      output_size=64,
                      use_self_attn=True,
                      norm='none',
                      model_capacity='standard'):
        coef = 2 if model_capacity == "lite" else 1
        upscale_block = upscale_nn if model_capacity == "lite" else upscale_ps
        activ_map_size = input_size
        use_norm = False if norm == 'none' else True

        inp = Input(shape=(input_size, input_size, nc_in))
        x = inp
        x = upscale_block(x, 256 // coef, use_norm, norm=norm)
        x = upscale_block(x, 128 // coef, use_norm, norm=norm)
        x = self_attn_block(x, 128 // coef) if use_self_attn else x
        x = upscale_block(x, 64 // coef, use_norm, norm=norm)
        x = res_block(x, 64 // coef, norm=norm)
        x = self_attn_block(x, 64 // coef) if use_self_attn else conv_block(x, 64 // coef, strides=1)

        outputs = []
        activ_map_size = activ_map_size * 8
        while activ_map_size < output_size:
            outputs.append(Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x))
            x = upscale_block(x, 64 // coef, use_norm, norm=norm)
            x = conv_block(x, 64 // coef, strides=1)
            activ_map_size *= 2
            pass

        alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
        bgr = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
        out = concatenate([alpha, bgr])
        outputs.append(out)
        return Model(inp, outputs)

    @staticmethod
    def build_discriminator(nc_in,
                            input_size=64,
                            use_self_attn=True,
                            norm='none') -> Model:
        activ_map_size = input_size
        use_norm = False if (norm == 'none') else True

        inp = Input(shape=(input_size, input_size, nc_in))
        x = conv_block_d(inp, 64, False)
        x = conv_block_d(x, 128, use_norm, norm=norm)
        x = conv_block_d(x, 256, use_norm, norm=norm)
        x = self_attn_block(x, 256) if use_self_attn else x

        activ_map_size = activ_map_size // 8
        while activ_map_size > 8:
            x = conv_block_d(x, 256, use_norm, norm=norm)
            x = self_attn_block(x, 256) if use_self_attn else x
            activ_map_size = activ_map_size // 2
            pass

        # use_bias should be True
        out = Conv2D(1, kernel_size=4, use_bias=False, padding="same")(x)
        return Model(inputs=[inp], outputs=out)

    @staticmethod
    def define_variables(net_gen):
        distorted_input = net_gen.inputs[0]
        fake_output = net_gen.outputs[-1]
        alpha = Lambda(lambda x: x[:, :, :, :1])(fake_output)
        bgr = Lambda(lambda x: x[:, :, :, 1:])(fake_output)

        masked_fake_output = alpha * bgr + (1 - alpha) * distorted_input

        fn_generate = K.function([distorted_input], [masked_fake_output])
        fn_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])])
        fn_abgr = K.function([distorted_input], [concatenate([alpha, bgr])])
        fn_bgr = K.function([distorted_input], [bgr])
        return distorted_input, fake_output, alpha, fn_generate, fn_mask, fn_abgr, fn_bgr

    def build_train_functions(self, loss_weights=None, **loss_config):
        assert loss_weights is not None, "loss weights are not provided."

        # Adversarial loss
        loss_disc_src, loss_adv_gen_src = adversarial_loss(
            self.net_disc_src, self.real_src, self.fake_src,
            self.distorted_src,
            loss_config["gan_training"],
            **loss_weights
        )
        loss_disc_dst, loss_adv_gen_dst = adversarial_loss(
            self.net_disc_dst, self.real_dst, self.fake_dst,
            self.distorted_dst,
            loss_config["gan_training"],
            **loss_weights
        )

        # Reconstruction loss
        loss_recon_gen_src = reconstruction_loss(
            self.real_src, self.fake_src,
            self.mask_eyes_src, self.net_gen_src.outputs,
            **loss_weights
        )
        loss_recon_gen_dst = reconstruction_loss(
            self.real_dst, self.fake_dst,
            self.mask_eyes_dst, self.net_gen_dst.outputs,
            **loss_weights
        )

        # Edge loss
        loss_edge_gen_src = edge_loss(self.real_src, self.fake_src, self.mask_eyes_src, **loss_weights)
        loss_edge_gen_dst = edge_loss(self.real_dst, self.fake_dst, self.mask_eyes_dst, **loss_weights)

        if loss_config['use_PL']:
            loss_pl_gen_src = perceptual_loss(
                self.real_src, self.fake_src, self.distorted_src,
                self.mask_eyes_src, self.vggface_feats, **loss_weights
            )
            loss_pl_gen_dst = perceptual_loss(
                self.real_dst, self.fake_dst, self.distorted_dst,
                self.mask_eyes_dst, self.vggface_feats, **loss_weights
            )
            pass
        else:
            loss_pl_gen_src = loss_pl_gen_dst = K.zeros(1)
            pass

        loss_gen_src = loss_adv_gen_src + loss_recon_gen_src + loss_edge_gen_src + loss_pl_gen_src
        loss_gen_dst = loss_adv_gen_dst + loss_recon_gen_dst + loss_edge_gen_dst + loss_pl_gen_dst

        # The following losses are rather trivial, thus their wegihts are fixed.
        # Cycle consistency loss
        if loss_config['use_cyclic_loss']:
            loss_gen_src += 10 * cyclic_loss(self.net_gen_src, self.net_gen_dst, self.real_src)
            loss_gen_dst += 10 * cyclic_loss(self.net_gen_dst, self.net_gen_src, self.real_dst)
            pass

        # Alpha mask loss
        if not loss_config['use_mask_hinge_loss']:
            loss_gen_src += 1e-2 * K.mean(K.abs(self.mask_src))
            loss_gen_dst += 1e-2 * K.mean(K.abs(self.mask_dst))
            pass
        else:
            loss_gen_src += 0.1 * K.mean(K.maximum(0., loss_config['m_mask'] - self.mask_src))
            loss_gen_dst += 0.1 * K.mean(K.maximum(0., loss_config['m_mask'] - self.mask_dst))
            pass

        # Alpha mask total variation loss
        loss_gen_src += 0.1 * K.mean(first_order(self.mask_src, axis=1))
        loss_gen_src += 0.1 * K.mean(first_order(self.mask_src, axis=2))
        loss_gen_dst += 0.1 * K.mean(first_order(self.mask_dst, axis=1))
        loss_gen_dst += 0.1 * K.mean(first_order(self.mask_dst, axis=2))

        # L2 weight decay
        # https://github.com/keras-team/keras/issues/2662
        for loss_tensor in self.net_gen_src.losses:
            loss_gen_src += loss_tensor
            pass
        for loss_tensor in self.net_gen_dst.losses:
            loss_gen_dst += loss_tensor
            pass
        for loss_tensor in self.net_disc_src.losses:
            loss_disc_src += loss_tensor
            pass
        for loss_tensor in self.net_disc_dst.losses:
            loss_disc_dst += loss_tensor
            pass

        weights_disc_src = self.net_disc_src.trainable_weights
        weights_gen_src = self.net_gen_src.trainable_weights
        weights_disc_dst = self.net_disc_dst.trainable_weights
        weights_gen_dst = self.net_gen_dst.trainable_weights

        # Define training functions
        training_updates = Adam(
            lr=self.learning_rate_disc * loss_config['lr_factor'], beta_1=0.5
        ).get_updates(loss_disc_src, weights_disc_src)

        self.net_disc_train_src = K.function(
            [self.distorted_src, self.real_src],
            [loss_disc_src],
            training_updates
        )
        training_updates = Adam(
            lr=self.learning_rate_gen * loss_config['lr_factor'], beta_1=0.5
        ).get_updates(loss_gen_src, weights_gen_src)

        self.net_gen_train_src = K.function(
            [self.distorted_src, self.real_src, self.mask_eyes_src],
            [loss_gen_src, loss_adv_gen_src, loss_recon_gen_src, loss_edge_gen_src,
             loss_pl_gen_src],
            training_updates
        )

        training_updates = Adam(
            lr=self.learning_rate_disc * loss_config['lr_factor'], beta_1=0.5
        ).get_updates(loss_disc_dst, weights_disc_dst)

        self.net_disc_train_dst = K.function(
            [self.distorted_dst, self.real_dst],
            [loss_disc_dst],
            training_updates
        )

        training_updates = Adam(
            lr=self.learning_rate_gen * loss_config['lr_factor'], beta_1=0.5
        ).get_updates(loss_gen_dst, weights_gen_dst)

        self.net_gen_train_dst = K.function(
            [self.distorted_dst, self.real_dst, self.mask_eyes_dst],
            [loss_gen_dst, loss_adv_gen_dst, loss_recon_gen_dst, loss_edge_gen_dst, loss_pl_gen_dst],
            training_updates
        )

    def build_pl_model(self, vggface_model, before_activ=False):
        # Define Perceptual Loss Model
        vggface_model.trainable = False
        if not before_activ:
            out_size112 = vggface_model.layers[1].output
            out_size55 = vggface_model.layers[36].output
            out_size28 = vggface_model.layers[78].output
            out_size7 = vggface_model.layers[-2].output
            pass
        else:
            out_size112 = vggface_model.layers[15].output  # misnamed: the output size is 55
            out_size55 = vggface_model.layers[35].output
            out_size28 = vggface_model.layers[77].output
            out_size7 = vggface_model.layers[-3].output
            pass

        self.vggface_feats = Model(vggface_model.input, [out_size112, out_size55, out_size28, out_size7])
        self.vggface_feats.trainable = False
        pass

    def load_weights(self, path="./models"):
        try:
            self.encoder.load_weights(f"{path}/encoder.h5")
            self.decoder_src.load_weights(f"{path}/decoder_A.h5")
            self.decoder_dst.load_weights(f"{path}/decoder_B.h5")
            self.net_disc_src.load_weights(f"{path}/netDA.h5")
            self.net_disc_dst.load_weights(f"{path}/netDB.h5")
            print("Файлы весов моделей успешно загружены.")
            pass
        except:
            print("Возникла ошибка при загрузке файлов весов.")
            pass

    def save_weights(self, path="./models"):
        try:
            self.encoder.save_weights(f"{path}/encoder.h5")
            self.decoder_src.save_weights(f"{path}/decoder_A.h5")
            self.decoder_dst.save_weights(f"{path}/decoder_B.h5")
            self.net_disc_src.save_weights(f"{path}/netDA.h5")
            self.net_disc_dst.save_weights(f"{path}/netDB.h5")
            print(f"Файлы весов моделей были сохранены в {path}.")
            pass
        except:
            print("Возникла ошибка при сохранении файлов весов.")
            pass

    def train_one_batch_gen(self, data_src, data_dst):
        if len(data_src) == 4 and len(data_dst) == 4:
            _, warped_src, target_src, bm_eyes_src = data_src
            _, warped_dst, target_dst, bm_eyes_dst = data_dst
            pass
        elif len(data_src) == 3 and len(data_dst) == 3:
            warped_src, target_src, bm_eyes_src = data_src
            warped_dst, target_dst, bm_eyes_dst = data_dst
            pass
        else:
            raise ValueError("Что-то не так с генератором входных данных.")
        err_gen_src = self.net_gen_train_src([warped_src, target_src, bm_eyes_src])
        err_gen_dst = self.net_gen_train_dst([warped_dst, target_dst, bm_eyes_dst])
        return err_gen_src, err_gen_dst

    def train_one_batch_disc(self, data_src, data_dst):
        if len(data_src) == 4 and len(data_dst) == 4:
            _, warped_src, target_src, _ = data_src
            _, warped_dst, target_dst, _ = data_dst
            pass
        elif len(data_src) == 3 and len(data_dst) == 3:
            warped_src, target_src, _ = data_src
            warped_dst, target_dst, _ = data_dst
            pass
        else:
            raise ValueError("Что-то не так с генератором входных данных.")
        err_disc_src = self.net_disc_train_src([warped_src, target_src])
        err_disc_dst = self.net_disc_train_dst([warped_dst, target_dst])
        return err_disc_src, err_disc_dst

    def transform_src2dst(self, img):
        return self.path_abgr_dst([[img]])

    def transform_dst2src(self, img):
        return self.path_abgr_src([[img]])

    pass


if __name__ == '__main__':
    K.set_learning_phase(1)
    # Number of CPU cores
    num_cpus = os.cpu_count()

    # Input/Output resolution
    RESOLUTION = 64  # 64x64, 128x128, 256x256
    assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, or 256."

    batch_size = 4

    # Use motion blur (data augmentation)
    # set True if training data contains images extracted from videos
    use_da_motion_blur = False

    # Use eye-aware training
    # require images generated from prep_binary_masks.ipynb
    use_bm_eyes = True

    # Probability of random color matching (data augmentation)
    prob_random_color_match = 0.5

    da_config = {
        "prob_random_color_match": prob_random_color_match,
        "use_da_motion_blur": use_da_motion_blur,
        "use_bm_eyes": use_bm_eyes
    }

    # Path to training images
    img_dir_src = './face_src/rgb'  # source face
    img_dir_dst = './face_dst/rgb'  # target face
    img_dir_src_bm_eyes = "./face_src/binary_mask"
    img_dir_dst_bm_eyes = "./face_dst/binary_mask"

    # Path to saved model weights
    models_dir = "./models"

    # Architecture configuration
    arch_config = {
        "IMAGE_SHAPE": (RESOLUTION, RESOLUTION, 3),
        "use_self_attn": True,
        "norm": "hybrid",
        "model_capacity": "lite"
    }

    # Loss function weights configuration
    loss_weights = {
        "w_D": 0.1,
        "w_recon": 1.,
        "w_edge": 0.1,
        "w_eyes": 30.,
        "w_pl": (0.01, 0.1, 0.3, 0.1)
    }

    # Init. loss config.
    loss_config = {
        "gan_training": "mixup_LSGAN",
        "use_PL": False,
        "PL_before_activ": True,
        "use_mask_hinge_loss": False,
        "m_mask": 0.,
        "lr_factor": 1.,
        "use_cyclic_loss": False
    }
    model = FaceswapModel(**arch_config)
    pass
