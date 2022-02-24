import glob
import os
import shutil
import time

from keras import backend as K

from converter import Converter
from converter.config import (
    ColorCorrectionType,
    ConverterConfig,
    TransformDirection,
    ImageOutputType,
    SegmentationType
)

TRAIN_DIR = "/home/alvin/faceswap_train"

# Path to saved model weights
MODELS_DIR = f"{TRAIN_DIR}/models"
VGGFACE_WEIGHT_FILE = f"{TRAIN_DIR}/models/vggface/rcmalli_vggface_tf_notop_resnet50.h5"

# Path to training images
SAVE_PATH_SOURCE = f"{TRAIN_DIR}/face_src"
SAVE_PATH_TARGET = f"{TRAIN_DIR}/face_dst"

PREFIX = "processed"

# 64x64, 128x128, 256x256
RESOLUTION = 64

fn_source_video = "../samples/source.mp4"
fn_target_video = "../samples/target.mp4"


def convert(config: ConverterConfig, arch_config: dict, duration=None):
    assert (arch_config['image_shape'][0] % 64) == 0, "RESOLUTION should be 64, 128, or 256."

    for d in glob.glob(f"{PREFIX}/frames/*"):
        shutil.rmtree(d)

    if config.direction == TransformDirection.AtoB:
        input_fn = fn_source_video
        output_fn = f"{PREFIX}/output_a2b.mp4"
    else:
        input_fn = fn_target_video
        output_fn = f"{PREFIX}/output_b2a.mp4"

    try:
        os.unlink(output_fn)
    except IOError:
        pass

    Converter(
        "../mtcnn_weights/", arch_config, model_path=MODELS_DIR,
        session=K.get_session(), config=config
    ).convert(
        input_fn, output_fn, duration=duration
    )

    pass


if __name__ == '__main__':
    time0 = time.time()

    min_face = 65

    # Architecture configuration
    arch_config = {
        "image_shape": (RESOLUTION, RESOLUTION, 3),
        "use_self_attn": True,
        "norm": "hybrid",  # instancenorm, batchnorm, layernorm, groupnorm, none, hybrid
        "model_capacity": "lite"
    }

    config = ConverterConfig(
        image_shape=arch_config['image_shape'],
        use_smoothed_bbox=True,
        use_kalman_filter=True,
        use_auto_downscaling=False,
        bbox_moving_avg_coef=0.65,
        min_face_area=min_face * min_face,
        kf_noise_coef=1e-3,
        color_correction=ColorCorrectionType.NONE,
        detection_threshold=0.8,
        roi_coverage=0.9,
        output_type=ImageOutputType.COMBINED,
        direction=TransformDirection.AtoB,
        segmentation=SegmentationType.EYES_ONLY
    )
    duration = (0, 10)

    convert(config, arch_config, duration)

    print("\nFinished in %.2f sec.\n" % (time.time() - time0))
    pass
