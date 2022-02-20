import numpy as np
from moviepy.editor import VideoFileClip

from .face_transformer import FaceTransformer
from .kalman_filter import KalmanFilter
from .landmarks_alignment import (
    get_src_landmarks,
    get_tar_landmarks,
    landmarks_match_mtcnn
)
from .options import Options as VCOptions, ImageOutputType
from .utils import (
    get_init_mask_map,
    get_init_comb_img,
    get_init_triple_img
)

KALMAN_FILTER_LOOP = 200


class VideoConverter:
    """
    Attributes:
        _face_transformer: FaceTransformer instance
        _face_detector: MTCNNFaceDetector instance
        prev_x0, prev_x1, prev_y0, prev_y1, frames: Variables for smoothing bounding box
        _kalman_filter0, _kalman_filter1: KalmanFilter instances for smoothing bounding box
    """

    def __init__(self):
        # Variables for smoothing bounding box
        self.prev_x0 = 0
        self.prev_x1 = 0
        self.prev_y0 = 0
        self.prev_y1 = 0
        self._frames = 0
        self._face_transformer = FaceTransformer()
        self._face_detector = None
        self._kalman_filter0 = None
        self._kalman_filter1 = None
        pass

    @property
    def gan_model(self):
        return self._face_transformer.model

    @gan_model.setter
    def gan_model(self, gm):
        self._face_transformer.model = gm
        pass

    @property
    def face_detector(self):
        return self._face_detector

    @face_detector.setter
    def face_detector(self, fd):
        self._face_detector = fd
        pass

    def convert(self, input_fn, output_fn, options, duration=None):
        """
        Parameters
        ----------
        input_fn : str
         Input file name
        output_fn : str
         Output file name
        options : VCOptions
         Options
        duration : None or tuple[int, int]
         Duration of video if specified

        Returns
        -------

        """

        if options.use_kalman_filter:
            self._init_kalman_filters(options.kf_noise_coef)

        self._frames = 0
        self.prev_x0 = self.prev_x1 = self.prev_y0 = self.prev_y1 = 0

        if self.face_detector is None:
            raise Exception(f"face detector has not been set through VideoConverter.set_face_detector() yet.")

        clip1 = VideoFileClip(input_fn)
        if type(duration) is tuple:
            clip = clip1.fl_image(lambda img: self.process_video(img, options)).subclip(duration[0], duration[1])
        else:
            clip = clip1.fl_image(lambda img: self.process_video(img, options))
        clip.write_videofile(output_fn, audio=True)
        clip1.reader.close()
        try:
            clip1.audio.reader.close_proc()
        except IOError as ioe:
            print(ioe)
            pass
        pass

    def process_video(self, input_img, options: VCOptions):
        """
        Transform detected faces in single input frame.

        Parameters
        ------
        input_img : ndarray
         Input image from video frame

        options : VCOptions
         Options for processing
        """
        image = input_img

        # detect face using MTCNN (faces: face bbox coord, pnts: landmarks coord.)
        faces, pnts = self.face_detector.detect_face(
            image, minsize=20,
            threshold=options.detection_threshold,
            factor=0.709,
            use_auto_downscaling=options.use_auto_downscaling,
            min_face_area=options.min_face_area
        )

        # check if any face detected
        if len(faces) == 0:
            triple_img = get_init_triple_img(input_img, no_face=True)
            pass

        # init. output image
        best_conf_score = 0
        comb_img = get_init_comb_img(input_img)
        mask_map = get_init_mask_map(image)

        # loop through all detected faces
        for i, (x0, y1, x1, y0, conf_score) in enumerate(faces):
            lms = pnts[:, i:i + 1]
            # smoothe the bounding box
            if options.use_smoothed_bbox:
                if self._frames != 0 and conf_score >= best_conf_score:
                    x0, x1, y0, y1 = self._get_smoothed_coord(
                        x0, x1, y0, y1,
                        img_shape=image.shape,
                        use_kalman_filter=options.use_kalman_filter,
                        ratio=options.bbox_moving_avg_coef,
                    )
                    self._set_prev_coord(x0, x1, y0, y1)
                    best_conf_score = conf_score
                    self._frames += 1
                elif conf_score <= best_conf_score:
                    self._frames += 1
                    pass
                else:
                    if conf_score >= best_conf_score:
                        self._set_prev_coord(x0, x1, y0, y1)
                        best_conf_score = conf_score
                    if options.use_kalman_filter:
                        for j in range(KALMAN_FILTER_LOOP):
                            self._kalman_filter0.predict()
                            self._kalman_filter1.predict()
                    self._frames += 1
                    pass
                pass

            # transform face
            try:
                # get detected face
                det_face_im = input_img[int(x0):int(x1), int(y0):int(y1), :]

                # get src/tar landmarks
                src_landmarks = get_src_landmarks(x0, x1, y0, y1, lms)
                tar_landmarks = get_tar_landmarks(det_face_im)

                # align detected face
                aligned_det_face_im = landmarks_match_mtcnn(det_face_im, src_landmarks, tar_landmarks)

                # face transform
                r_im, r_rgb, r_a = self._face_transformer.transform(
                    aligned_det_face_im,
                    direction=options.direction,
                    roi_coverage=options.roi_coverage,
                    color_correction=options.color_correction,
                    image_shape=options.image_shape
                )

                # reverse alignment
                rev_aligned_det_face_im = landmarks_match_mtcnn(r_im, tar_landmarks, src_landmarks)
                rev_aligned_det_face_im_rgb = landmarks_match_mtcnn(r_rgb, tar_landmarks, src_landmarks)
                rev_aligned_mask = landmarks_match_mtcnn(r_a, tar_landmarks, src_landmarks)

                # merge source face and transformed face
                result = np.zeros_like(det_face_im)
                result = rev_aligned_mask / 255 * rev_aligned_det_face_im_rgb + (
                        1 - rev_aligned_mask / 255) * det_face_im
                result_a = rev_aligned_mask
            except:
                # catch exceptions for landmarks alignment errors (if any)
                print(f"Face alignment error occurs at frame {self._frames}.")
                # get detected face
                det_face_im = input_img[int(x0):int(x1), int(y0):int(y1), :]

                result, _, result_a = self._face_transformer.transform(
                    det_face_im,
                    direction=options.direction,
                    roi_coverage=options.roi_coverage,
                    color_correction=options.color_correction,
                    image_shape=options.image_shape
                )
                pass

            comb_img[
             int(x0):int(x1),
             input_img.shape[1] + int(y0):input_img.shape[1] + int(y1),
             :] = result

            # Enhance output
            if options.enhance != 0:
                comb_img = -1 * options.enhance * get_init_comb_img(input_img) + (1 + options.enhance) * comb_img
                comb_img = np.clip(comb_img, 0, 255)
                pass

            if conf_score >= best_conf_score:
                mask_map[int(x0):int(x1), int(y0):int(y1), :] = result_a
                mask_map = np.clip(mask_map + .15 * input_img, 0, 255)
                # Possible bug: when small faces are detected before the most confident face,
                #               the mask_map will show brighter input_img
                pass
            else:
                mask_map[int(x0):int(x1), int(y0):int(y1), :] += result_a
                mask_map = np.clip(mask_map, 0, 255)
                pass

            triple_img = get_init_triple_img(input_img)
            triple_img[:, :input_img.shape[1] * 2, :] = comb_img
            triple_img[:, input_img.shape[1] * 2:, :] = mask_map
            pass

        if options.output_type == ImageOutputType.SINGLE:
            return comb_img[:, input_img.shape[1]:, :]
        elif options.output_type == ImageOutputType.COMBINED:
            return comb_img
        elif options.output_type == ImageOutputType.TRIPLE:
            return triple_img
        pass

    def _get_smoothed_coord(self, x0, x1, y0, y1, img_shape, use_kalman_filter=True, ratio=0.65):
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
