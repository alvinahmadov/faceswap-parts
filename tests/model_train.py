import os

import keras.backend as K
from IPython.display import clear_output

from nn.faceswap_model import FaceswapModel
from nn.vggface import RESNET50

# Input/Output resolution
RESOLUTION = 64  # 64x64, 128x128, 256x256
TRAIN_DIR = "/home/alvin/faceswap_train"

# Path to saved model weights
MODELS_DIR = f"{TRAIN_DIR}/models"

# Path to training images
SAVE_PATH_SOURCE = f"{TRAIN_DIR}/face_src"
SAVE_PATH_TARGET = f"{TRAIN_DIR}/face_dst"

WEIGHTS_FILE = f"{MODELS_DIR}/rcmalli_vggface_tf_notop_resnet50.h5"


if __name__ == '__main__':
    clear_output()
    K.set_learning_phase(1)
    # Number of CPU cores
    num_cpus = os.cpu_count()

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
    img_dir_src = f"{SAVE_PATH_SOURCE}/rgb"  # source face
    img_dir_dst = f"{SAVE_PATH_TARGET}/rgb"  # target face
    img_dir_src_bm_eyes = f"{SAVE_PATH_SOURCE}/binary_mask"
    img_dir_dst_bm_eyes = f"{SAVE_PATH_TARGET}/binary_mask"
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

    vggface = RESNET50(include_top=False, weights=None, input_shape=(224, 224, 3))
    vggface.load_weights(WEIGHTS_FILE, by_name=True)

    model.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])
    model.build_train_functions(loss_weights=loss_weights, **loss_config)
    pass
