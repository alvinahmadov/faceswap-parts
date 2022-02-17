import glob
import shutil
import unittest
from pathlib import Path

import keras.backend as K

from detector import MTCNNFaceDetector
from preprocess import preprocess_video

MTCNN_MODEL_PATH = "../mtcnn_weights/"
SOURCE_SAVE_PATH = "./face_src"
TARGET_SAVE_PATH = "./face_dst"

fn_source_video = "../samples/source.mp4"
fn_target_video = "../samples/target.mp4"

SOURCE_FACES_CNT = 311
TARGET_FACES_CNT = 61


class TestFaceDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.save_interval = 5  # perform face detection every {save_interval} frames
        self.face_detector = MTCNNFaceDetector(sess=K.get_session(), model_path=MTCNN_MODEL_PATH)

        for pathname in [SOURCE_SAVE_PATH, TARGET_SAVE_PATH]:
            shutil.rmtree(pathname, ignore_errors=True)
            Path(pathname).mkdir(parents=True, exist_ok=True)
            pass
        pass

    def tearDown(self) -> None:
        del self.face_detector
        shutil.rmtree('dummy.mp4')
        pass

    def test_detect_face(self):
        preprocess_video(fn_source_video, self.face_detector, self.save_interval, f"{SOURCE_SAVE_PATH}/")
        preprocess_video(fn_target_video, self.face_detector, self.save_interval, f"{TARGET_SAVE_PATH}/")

        face_src_glob_len = len(glob.glob(f"{SOURCE_SAVE_PATH}/rgb/*.*"))
        face_dst_glob_len = len(glob.glob(f"{TARGET_SAVE_PATH}/rgb/*.*"))

        self.assertEqual(
            face_src_glob_len, SOURCE_FACES_CNT,
            f"Extracted faces length ({face_src_glob_len}) doesn't match actual "
            f"faces ({SOURCE_FACES_CNT}) from source video: {fn_source_video}."
        )

        self.assertEqual(
            face_dst_glob_len, TARGET_FACES_CNT,
            f"Extracted faces length ({face_dst_glob_len}) doesn't match actual "
            f"faces ({TARGET_FACES_CNT}) from target video: {fn_target_video}."
        )
        pass
    pass
