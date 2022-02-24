import keras.backend as K
from keras import Input
from keras.layers import Lambda, concatenate
from keras.models import Model
from keras.optimizers import Adam

from converter.config import TransformDirection
from nn.losses import (
    first_order, cyclic_loss, adversarial_loss,
    reconstruction_loss, edge_loss, perceptual_loss
)
from nn import Encoder, Decoder, Discriminator


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

    fn_encoder = "encoder.h5"
    fn_decoder_a = "decoder_A.h5"
    fn_decoder_b = "decoder_B.h5"
    fn_disc_a = "netDA.h5"
    fn_disc_b = "netDB.h5"

    def __init__(self, **arch_config):
        """
        Parameters
        ----------
        arch_config : dict
         A dictionary that contains architecture configurations.
        """
        self.num_gen_input_channels = 3
        self.num_disc_input_channels = 6
        self.image_shape = arch_config['image_shape']
        self.learning_rate_disc = 2e-4
        self.learning_rate_gen = 1e-4
        self.use_self_attn = arch_config['use_self_attn']
        self.norm = arch_config['norm']
        self.model_capacity = arch_config['model_capacity']
        self.enc_nc_out = 256 if self.model_capacity == "lite" else 512

        self.net_disc_train_a = None
        self.net_disc_train_b = None

        self.net_gen_train_a = None
        self.net_gen_train_b = None

        self.vggface_feats = None
        self.weights_loaded = False

        # define networks
        # autocoders
        im_shape_x = self.image_shape[0]

        encoder_shape = (im_shape_x, im_shape_x, self.num_gen_input_channels)
        decoder_shape = (8, 8, self.enc_nc_out)
        disc_shape = (im_shape_x, im_shape_x, self.num_disc_input_channels)

        self.encoder = Encoder(self.model_capacity,
                               self.image_shape[0],
                               self.use_self_attn,
                               self.norm).build(encoder_shape)

        self.decoder_a = Decoder(self.model_capacity, 8, self.image_shape[0],
                                 self.use_self_attn,
                                 self.norm).build(decoder_shape)

        self.decoder_b = Decoder(self.model_capacity, 8, self.image_shape[0],
                                 self.use_self_attn,
                                 self.norm).build(decoder_shape)

        self.disc_a = Discriminator(self.image_shape[0], self.use_self_attn, self.norm).build(disc_shape)

        self.disc_b = Discriminator(self.image_shape[0], self.use_self_attn, self.norm).build(disc_shape)

        inputs = Input(shape=self.image_shape)

        # init generator networks
        self.gen_a = Model(inputs, self.decoder_a(self.encoder(inputs)))
        self.gen_b = Model(inputs, self.decoder_b(self.encoder(inputs)))

        # define variables
        self.distorted_a, self.fake_a, self.mask_a, \
            self.path_a, self.path_mask_a, self.path_abgr_a, \
            self.path_bgr_a = self.define_variables(self.gen_a)

        self.distorted_b, self.fake_b, self.mask_b, \
            self.path_b, self.path_mask_b, self.path_abgr_b, \
            self.path_bgr_b = self.define_variables(self.gen_b)

        self.real_a = Input(shape=self.image_shape)
        self.real_b = Input(shape=self.image_shape)

        self.mask_eyes_a = Input(shape=self.image_shape)
        self.mask_eyes_b = Input(shape=self.image_shape)
        pass

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
        return (distorted_input, fake_output, alpha,
                fn_generate, fn_mask, fn_abgr, fn_bgr)

    def build_train_functions(self, loss_weights=None, **loss_config):
        assert loss_weights is not None, "loss weights are not provided."

        # Adversarial loss
        loss_disc_a, loss_adv_gen_a = adversarial_loss(
            self.disc_a, self.real_a, self.fake_a,
            self.distorted_a,
            loss_config["gan_training"],
            **loss_weights
        )

        loss_disc_b, loss_adv_gen_b = adversarial_loss(
            self.disc_b, self.real_b, self.fake_b,
            self.distorted_b,
            loss_config["gan_training"],
            **loss_weights
        )

        # Reconstruction loss
        loss_recon_gen_a = reconstruction_loss(
            self.real_a, self.fake_a,
            self.mask_eyes_a, self.gen_a.outputs,
            **loss_weights
        )

        loss_recon_gen_b = reconstruction_loss(
            self.real_b, self.fake_b,
            self.mask_eyes_b, self.gen_b.outputs,
            **loss_weights
        )

        # Edge loss
        loss_edge_gen_a = edge_loss(self.real_a, self.fake_a, self.mask_eyes_a, **loss_weights)
        loss_edge_gen_b = edge_loss(self.real_b, self.fake_b, self.mask_eyes_b, **loss_weights)

        if loss_config['use_PL']:
            loss_pl_gen_a = perceptual_loss(
                self.real_a, self.fake_a, self.distorted_a,
                self.mask_eyes_a, self.vggface_feats, **loss_weights
            )

            loss_pl_gen_b = perceptual_loss(
                self.real_b, self.fake_b, self.distorted_b,
                self.mask_eyes_b, self.vggface_feats, **loss_weights
            )
            pass
        else:
            loss_pl_gen_a = loss_pl_gen_b = K.zeros(1)
            pass

        loss_gen_a = loss_adv_gen_a + loss_recon_gen_a + loss_edge_gen_a + loss_pl_gen_a
        loss_gen_b = loss_adv_gen_b + loss_recon_gen_b + loss_edge_gen_b + loss_pl_gen_b

        # The following losses are rather trivial, thus their wegihts are fixed.
        # Cycle consistency loss
        if loss_config['use_cyclic_loss']:
            loss_gen_a += 10 * cyclic_loss(self.gen_a, self.gen_b, self.real_a)
            loss_gen_b += 10 * cyclic_loss(self.gen_b, self.gen_a, self.real_b)
            pass

        # Alpha mask loss
        if not loss_config['use_mask_hinge_loss']:
            loss_gen_a += 1e-2 * K.mean(K.abs(self.mask_a))
            loss_gen_b += 1e-2 * K.mean(K.abs(self.mask_b))
            pass
        else:
            loss_gen_a += 0.1 * K.mean(K.maximum(0., loss_config['m_mask'] - self.mask_a))
            loss_gen_b += 0.1 * K.mean(K.maximum(0., loss_config['m_mask'] - self.mask_b))
            pass

        # Alpha mask total variation loss
        loss_gen_a += 0.1 * K.mean(first_order(self.mask_a, axis=1))
        loss_gen_a += 0.1 * K.mean(first_order(self.mask_a, axis=2))
        loss_gen_b += 0.1 * K.mean(first_order(self.mask_b, axis=1))
        loss_gen_b += 0.1 * K.mean(first_order(self.mask_b, axis=2))

        # L2 weight decay
        # https://github.com/keras-team/keras/issues/2662
        for loss_tensor in self.gen_a.losses:
            loss_gen_a += loss_tensor
            pass
        for loss_tensor in self.gen_b.losses:
            loss_gen_b += loss_tensor
            pass
        for loss_tensor in self.disc_a.losses:
            loss_disc_a += loss_tensor
            pass
        for loss_tensor in self.disc_b.losses:
            loss_disc_b += loss_tensor
            pass

        weights_disc_a = self.disc_a.trainable_weights
        weights_gen_a = self.gen_a.trainable_weights
        weights_disc_b = self.disc_b.trainable_weights
        weights_gen_b = self.gen_b.trainable_weights

        # Define training functions
        training_updates = Adam(
            lr=self.learning_rate_disc * loss_config['lr_factor'], beta_1=0.5
        ).get_updates(loss_disc_a, weights_disc_a)

        self.net_disc_train_a = K.function(
            [self.distorted_a, self.real_a],
            [loss_disc_a],
            updates=training_updates
        )

        training_updates = Adam(
            lr=self.learning_rate_gen * loss_config['lr_factor'], beta_1=0.5
        ).get_updates(loss_gen_a, weights_gen_a)

        self.net_gen_train_a = K.function(
            [self.distorted_a, self.real_a, self.mask_eyes_a],
            [loss_gen_a, loss_adv_gen_a, loss_recon_gen_a, loss_edge_gen_a, loss_pl_gen_a],
            updates=training_updates
        )

        training_updates = Adam(
            lr=self.learning_rate_disc * loss_config['lr_factor'], beta_1=0.5
        ).get_updates(loss_disc_b, weights_disc_b)

        self.net_disc_train_b = K.function(
            [self.distorted_b, self.real_b], [loss_disc_b], updates=training_updates
        )

        training_updates = Adam(
            lr=self.learning_rate_gen * loss_config['lr_factor'], beta_1=0.5
        ).get_updates(loss_gen_b, weights_gen_b)

        self.net_gen_train_b = K.function(
            [self.distorted_b, self.real_b, self.mask_eyes_b],
            [loss_gen_b, loss_adv_gen_b, loss_recon_gen_b, loss_edge_gen_b, loss_pl_gen_b],
            updates=training_updates
        )
        pass

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

        self.vggface_feats = Model(
            vggface_model.input,
            [out_size112, out_size55, out_size28, out_size7]
        )
        self.vggface_feats.trainable = False
        pass

    def load_weights(self, path="./models"):
        try:
            self.encoder.load_weights(f"{path}/{self.fn_encoder}")
            self.decoder_a.load_weights(f"{path}/{self.fn_decoder_a}")
            self.decoder_b.load_weights(f"{path}/{self.fn_decoder_b}")
            self.disc_a.load_weights(f"{path}/{self.fn_disc_a}")
            self.disc_b.load_weights(f"{path}/{self.fn_disc_b}")
            self.weights_loaded = True
            print("Файлы весов модели успешно загружены.")
            pass
        except FileNotFoundError as fe:
            print(f"Файл весов не найден.\n{fe}")
            pass
        except IOError as ioe:
            print(f"Произошла ошибка во время загрузки весов.\n{ioe}")
            pass
        pass

    def save_weights(self, path="./models"):

        try:
            self.encoder.save_weights(f"{path}/{self.fn_encoder}")
            self.decoder_a.save_weights(f"{path}/{self.fn_decoder_a}")
            self.decoder_b.save_weights(f"{path}/{self.fn_decoder_b}")
            self.disc_a.save_weights(f"{path}/{self.fn_disc_a}")
            self.disc_b.save_weights(f"{path}/{self.fn_disc_b}")
            print(f"Файлы весов модели были успешно сохранены в `{path}`.")
            pass
        except FileNotFoundError as fne:
            print("Файл весов не удалось сохранить.\nErr: " + str(fne))
        except IOError as e:
            print("Произошла ошибка во время сохранения весов.\nErr: " + str(e))
            pass
        pass

    def train_one_batch_gen(self, data_a, data_b):
        if len(data_a) == 4 and len(data_b) == 4:
            _, warped_a, target_a, bm_eyes_a = data_a
            _, warped_b, target_b, bm_eyes_b = data_b
            pass
        elif len(data_a) == 3 and len(data_b) == 3:
            warped_a, target_a, bm_eyes_a = data_a
            warped_b, target_b, bm_eyes_b = data_b
            pass
        else:
            raise ValueError("Something's wrong with the input data generator.")
            pass
        err_gen_a = self.net_gen_train_a([warped_a, target_a, bm_eyes_a])
        err_gen_b = self.net_gen_train_b([warped_b, target_b, bm_eyes_b])
        return err_gen_a, err_gen_b

    def train_one_batch_disc(self, data_a, data_b):
        if len(data_a) == 4 and len(data_b) == 4:
            _, warped_a, target_a, _ = data_a
            _, warped_b, target_b, _ = data_b
            pass
        elif len(data_a) == 3 and len(data_b) == 3:
            warped_a, target_a, _ = data_a
            warped_b, target_b, _ = data_b
            pass
        else:
            raise ValueError("Something's wrong with the input data generator.")
        err_disc_a = self.net_disc_train_a([warped_a, target_a])
        err_disc_b = self.net_disc_train_b([warped_b, target_b])
        return err_disc_a, err_disc_b

    def transform(self, image, direction, method='abgr'):
        if direction == TransformDirection.AtoB:
            if method == "abgr":
                return self._transform_abgr_ab(image)
            elif method == "bgr":
                return self._transform_bgr_ab(image)
            else:
                raise ValueError(f"No such transform method `{method}`.")
            pass
        elif direction == TransformDirection.BtoA:
            if method == "abgr":
                return self._transform_abgr_ba(image)
            elif method == "bgr":
                return self._transform_bgr_ba(image)
            else:
                raise ValueError(f"No such transform method `{method}`.")
            pass
        else:
            raise ValueError(f"direction should be either AtoB or BtoA, recieved {direction}.")
        pass

    def _transform_abgr_ab(self, image):
        return self.path_abgr_b([[image]])

    def _transform_abgr_ba(self, image):
        return self.path_abgr_a([[image]])

    def _transform_bgr_ab(self, image):
        return self.path_bgr_b([[image]])

    def _transform_bgr_ba(self, image):
        return self.path_bgr_a([[image]])

    pass
