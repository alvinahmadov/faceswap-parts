import enum


class TransformDirection(enum.Enum):
    AtoB = 0,
    BtoA = 1
    pass


class ImageOutputType(enum.Enum):
    SINGLE = 0  # return only result image
    COMBINED = 1  # return input and result image combined as one
    TRIPLE = 3  # return input, result and mask heatmap image combined as one
    pass


class ColorCorrection(enum.Enum):
    ADAIN = 0
    ADAIN_XYZ = 1
    HISTMATCH = 2
    pass


class Options:
    def __init__(self, image_shape, use_smoothed_bbox=True, use_kalman_filter=True,
                 use_auto_downscaling=False, bbox_moving_avg_coef=0.65,
                 min_face_area=35 * 35, kf_noise_coef=1e-3, color_correction=ColorCorrection.HISTMATCH,
                 detection_threshold=0.8, roi_coverage=0.9, enhance=0.0,
                 output_type=ImageOutputType.TRIPLE, direction=TransformDirection.AtoB):
        self.image_shape = image_shape
        self.use_smoothed_bbox: bool = use_smoothed_bbox
        self.use_kalman_filter: bool = use_kalman_filter
        self.use_auto_downscaling: bool = use_auto_downscaling
        self.bbox_moving_avg_coef: float = bbox_moving_avg_coef
        self.min_face_area = min_face_area
        self.kf_noise_coef = kf_noise_coef
        self.color_correction: ColorCorrection = color_correction
        self.detection_threshold: float = detection_threshold
        self.roi_coverage: float = roi_coverage
        self.enhance: float = enhance
        self.output_type: ImageOutputType = output_type
        self.direction: TransformDirection = direction

        self._check_options()
        pass

    def _check_options(self):
        if self.roi_coverage <= 0 or self.roi_coverage >= 1:
            raise ValueError(f"roi_coverage should be between 0 and 1 (exclusive).")
        if self.bbox_moving_avg_coef < 0 or self.bbox_moving_avg_coef > 1:
            raise ValueError(f"bbox_moving_avg_coef should be between 0 and 1 (inclusive).")
        if self.detection_threshold < 0 or self.detection_threshold > 1:
            raise ValueError(f"detec_threshold should be between 0 and 1 (inclusive).")
        pass

    pass
