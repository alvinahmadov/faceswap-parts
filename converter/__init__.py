import logging
import os

import numpy as np
from moviepy.editor import VideoFileClip

from detector import MTCNNFaceDetector as Detector
from utils import DummyLogger
from .config import (
    ConverterConfig,
    ImageOutputType,
    ColorCorrectionType,
    TransformDirection,
    SegmentationType
)
from .kalman_filter import KalmanFilter
from .landmarks_alignment import (
    get_src_landmarks,
    get_tar_landmarks,
    landmarks_match_mtcnn
)
from .segmentation import FaceSegmentation
from .transformer import FaceTransformer
from .utils import (
    fillzeros,
    draw_landmarks,
    get_init_mask_map,
    get_init_comb_img,
    get_init_triple_img
)

KALMAN_FILTER_LOOP = 200

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 0))
logger = logging.getLogger(__name__) \
    if os.environ.get('ENABLE_LOGGING', False) else DummyLogger()

DEFAULT_CONFIG = ConverterConfig(
    image_shape=64,
    use_smoothed_bbox=True,
    use_kalman_filter=True,
    use_auto_downscaling=False,
    bbox_moving_avg_coef=0.65,
    min_face_area=35 * 35,
    kf_noise_coef=1e-3,
    color_correction=ColorCorrectionType.NONE,
    detection_threshold=0.8,
    roi_coverage=0.9,
    output_type=ImageOutputType.COMBINED,
    direction=TransformDirection.BtoA,
    segmentation=SegmentationType.ALL
)


class Converter:
    """
    Attributes:
        _face_transformer: FaceTransformer instance
        _face_detector: MTCNNFaceDetector instance
        prev_x0, prev_x1, prev_y0, prev_y1, frames: Variables for smoothing bounding box
        _kalman_filter0, _kalman_filter1: KalmanFilter instances for smoothing bounding box
    """

    def __init__(self, det_models_path, arch_config=None,
                 model_path=None, config=DEFAULT_CONFIG, session=None):
        logger.info(
            "Initializing Converter class with parameters: %s, %s, %s, %s" % (
                det_models_path, str(arch_config), model_path, str(session)
            ))
        # Variables for smoothing bounding box
        self.prev_x0 = 0
        self.prev_x1 = 0
        self.prev_y0 = 0
        self.prev_y1 = 0
        self._frames = 0
        self.config = config

        self._face_detector = Detector(session, det_models_path)
        self._face_transformer = FaceTransformer(arch_config, model_path) \
            if arch_config is not None else None

        self._face_segment = FaceSegmentation()

        self._kalman_filter0 = None
        self._kalman_filter1 = None
        pass

    @property
    def model(self):
        return self._face_transformer.model

    @model.setter
    def model(self, gm):
        self._face_transformer.model = gm
        pass

    @property
    def detector(self):
        return self._face_detector

    @detector.setter
    def detector(self, fd):
        self._face_detector = fd
        pass

    def convert(self, input_fn, output_fn, duration=None, audio=False):
        """
        Parameters
        ----------
        audio : bool
         Save audio from original video
        input_fn : str
         Input file name
        output_fn : str
         Output file name
        duration : None or tuple[int, int]
         Duration of video if specified

        Returns
        -------

        """
        logger.info(
            "Starting convert process for video frames from `%s` to `%s`\n and config %s." % (
                input_fn, output_fn, str(config),
            )
        )

        if self.config.use_kalman_filter:
            self._init_kalman_filters(self.config.kf_noise_coef)

        self._frames = 0
        self.prev_x0 = self.prev_x1 = self.prev_y0 = self.prev_y1 = 0

        if self.detector is None:
            logger.error("Face detector has not been initialized yet.")
            raise Exception

        try:
            clip1 = VideoFileClip(input_fn)
            if type(duration) is tuple:
                logger.info("Duration specified from %i to %i" % (duration[0], duration[1]))
                clip = clip1.fl_image(lambda img: self.process_video(img, self.config)).subclip(duration[0],
                                                                                                duration[1])
            else:
                clip = clip1.fl_image(lambda img: self.process_video(img, self.config))
            clip.write_videofile(output_fn, audio=audio)
            clip1.reader.close()
            clip1.audio.reader.close_proc()
        except Exception as e:
            logger.error("Error: %s" % e)
            pass
        pass

    def process_video(self, frame, config: ConverterConfig):
        """
        Transform detected faces in single input frame.

        Parameters
        ------
        frame : ndarray
         Input image from video frame

        config : ConverterConfig
         ConverterConfig for processing
        """
        # image = frame
        image_shape = self.model.image_shape \
            if config.image_shape is None else config.image_shape

        # detect face using MTCNN (faces: face bbox coord, pnts: landmarks coord.)
        faces, pnts = self.detector.detect_face(
            frame, minsize=20,
            threshold=config.detection_threshold,
            factor=0.709,
            use_auto_downscaling=config.use_auto_downscaling,
            min_face_area=config.min_face_area
        )

        # check if any face detected
        if len(faces) == 0:
            triple_img = get_init_triple_img(frame, no_face=True)

        logger.debug("Detected %i face(s) in a frame and point count %i" % (len(faces), len(pnts)))

        best_conf_score = 0
        # init. output image
        comb_img = get_init_comb_img(frame)
        mask_map = get_init_mask_map(frame)

        # loop through all detected faces
        for i, (x0, y1, x1, y0, conf_score) in enumerate(faces):
            logger.debug(
                "Face coordinates are X(%s, %s), Y(%s, %s) with confidence score %.2f" % (
                    x0, x1, y0, y1, conf_score
                )
            )
            lms = pnts[:, i:i + 1]
            # smoothe the bounding box
            if config.use_smoothed_bbox:
                logger.debug("Smoothing bounding box")
                if self._frames != 0 and conf_score >= best_conf_score:
                    logger.debug("cond: frames != 0 and conf_score >= best_conf_score")
                    x0, x1, y0, y1 = self._get_smoothed_coord(
                        x0, x1, y0, y1,
                        img_shape=frame.shape,
                        use_kalman_filter=config.use_kalman_filter,
                        ratio=config.bbox_moving_avg_coef,
                    )
                    self._set_prev_coord(x0, x1, y0, y1)
                    best_conf_score = conf_score
                    self._frames += 1
                elif conf_score <= best_conf_score:
                    logger.debug("cond: conf_score <= best_conf_score")
                    self._frames += 1
                    pass
                else:
                    if conf_score >= best_conf_score:
                        self._set_prev_coord(x0, x1, y0, y1)
                        best_conf_score = conf_score
                    if config.use_kalman_filter:
                        logger.debug("Using kalman filter")
                        for j in range(KALMAN_FILTER_LOOP):
                            self._kalman_filter0.predict()
                            self._kalman_filter1.predict()
                            pass
                    self._frames += 1
                    pass
                pass

            # transform face
            comb_img, triple_img = self._transform(
                frame, image_shape, face=(x0, x1, y0, y1, conf_score),
                config=config, comb_img=comb_img,
                mask_map=mask_map, lms=lms, face_idx=i + 1
            )
            pass

        if config.output_type == ImageOutputType.SINGLE:
            return comb_img[:, frame.shape[1]:, :]
        elif config.output_type == ImageOutputType.COMBINED:
            return comb_img
        elif config.output_type == ImageOutputType.TRIPLE:
            return triple_img
        pass

    def _transform(self, frame, image_shape, face,
                   config, comb_img, mask_map, lms, face_idx=0):
        """
        Here goes actual transformation for source and target faces
        Parameters
        ----------
        frame : ndarray
         Frame from video with target face
        image_shape
        face
        config
        comb_img
        mask_map
        lms
        face_idx

        Returns
        -------

        """
        logger.info("Starting face transformation for face %i" % face_idx)
        # init. output image
        best_conf_score = 0.0
        x0, x1, y0, y1, conf_score = face
        # get detected face
        detected_face_img = frame[int(x0):int(x1), int(y0):int(y1), :]

        try:
            # get src/tar landmarks
            src_landmarks = get_src_landmarks(x0, y0, lms, face_idx - 1)
            tar_landmarks = get_tar_landmarks(detected_face_img)

            # align detected face
            aligned_det_face_im = landmarks_match_mtcnn(detected_face_img, src_landmarks, tar_landmarks)

            # face transform
            r_im, r_rgb, r_alpha = self._face_transformer.transform(aligned_det_face_im,
                                                                    config.direction,
                                                                    config.roi_coverage,
                                                                    config.color_correction,
                                                                    image_shape)

            # reverse alignment
            rev_aligned_det_face_im_rgb = landmarks_match_mtcnn(r_rgb, tar_landmarks, src_landmarks)
            rev_aligned_mask = landmarks_match_mtcnn(r_alpha, tar_landmarks, src_landmarks)

            seg_im_rgb = self._face_segment.get_segments(
                detected_face_img, rev_aligned_det_face_im_rgb, config.segmentation_type
            )

            # merge source face and transformed face
            # TODO(alvin): Maybe needed to reverse masking for aligned_im_a and aligned_im_b
            #  (which does no swap).
            aligned_im_a: np.ndarray = (1 - rev_aligned_mask / 255) * seg_im_rgb
            aligned_im_b: np.ndarray = (rev_aligned_mask / 255) * detected_face_img

            # We merge original image and swapped image parts
            result = aligned_im_a + aligned_im_b
            result_a = rev_aligned_mask

        except Exception as ex:
            # catch exceptions for landmarks alignment errors (if any)
            logger.error(f"Face alignment error occurred at frame {self._frames}. "
                         f"Transforming without alignment."
                         f"\nError {ex}",
                         exc_info=True)

            result, _, result_a = self._face_transformer.transform(
                detected_face_img,
                direction=config.direction,
                roi_coverage=config.roi_coverage,
                color_correction=config.color_correction,
                image_shape=image_shape
            )
            pass

        # create image with original and swapped frames
        comb_img[int(x0):int(x1), frame.shape[1] + int(y0):frame.shape[1] + int(y1), :] = result

        # Enhance output
        if config.enhance != 0:
            comb_img = -1 * config.enhance * get_init_comb_img(frame) + (1 + config.enhance) * comb_img
            comb_img = np.clip(comb_img, 0, 255)
            pass

        if conf_score >= best_conf_score:
            mask_map[int(x0):int(x1), int(y0):int(y1), :] = result_a
            mask_map = np.clip(mask_map + .15 * frame, 0, 255)
            pass
        else:
            mask_map[int(x0):int(x1), int(y0):int(y1), :] += result_a
            mask_map = np.clip(mask_map, 0, 255)
            pass

        # create frame with original, swapped and mask frame
        triple_img = get_init_triple_img(frame)
        triple_img[:, :frame.shape[1] * 2, :] = comb_img
        triple_img[:, frame.shape[1] * 2:, :] = mask_map

        return comb_img, triple_img

    def _get_smoothed_coord(self, x0, x1, y0, y1, img_shape, use_kalman_filter=True, ratio=0.65):
        logger.debug(f"Smoothening face coordinates with ratio: {ratio}")
        if not use_kalman_filter:
            x0 = int(ratio * self.prev_x0 + (1 - ratio) * x0)
            x1 = int(ratio * self.prev_x1 + (1 - ratio) * x1)
            y1 = int(ratio * self.prev_y1 + (1 - ratio) * y1)
            y0 = int(ratio * self.prev_y0 + (1 - ratio) * y0)
        else:
            x0y0 = np.array([x0, y0]).astype(np.float32)
            x1y1 = np.array([x1, y1]).astype(np.float32)
            self._kalman_filter0.correct(x0y0)
            pred_x0y0 = self._kalman_filter0.predict()
            self._kalman_filter1.correct(x1y1)
            pred_x1y1 = self._kalman_filter1.predict()
            x0 = np.max([0, pred_x0y0[0][0]]).astype(np.int)
            x1 = np.min([img_shape[0], pred_x1y1[0][0]]).astype(np.int)
            y0 = np.max([0, pred_x0y0[1][0]]).astype(np.int)
            y1 = np.min([img_shape[1], pred_x1y1[1][0]]).astype(np.int)
            if x0 == x1 or y0 == y1:
                x0, y0, x1, y1 = self.prev_x0, self.prev_y0, self.prev_x1, self.prev_y1
        return x0, x1, y0, y1

    def _set_prev_coord(self, x0, x1, y0, y1):
        logger.debug("Setting previous coordinates X(%s, %s), Y(%s, %s)" % (
            x0, x1, y0, y1
        ))
        self.prev_x0 = x0
        self.prev_x1 = x1
        self.prev_y1 = y1
        self.prev_y0 = y0
        pass

    def _init_kalman_filters(self, noise_coef):
        self._kalman_filter0 = KalmanFilter(noise_coefficient=noise_coef)
        self._kalman_filter1 = KalmanFilter(noise_coefficient=noise_coef)
        pass

    pass
