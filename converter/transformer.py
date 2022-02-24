import cv2
import numpy as np

from nn import FaceswapModel
from .color_correction import adain, color_hist_match, ColorCorrectionType


class FaceTransformer:
    """
    Transforms face to face in image

    Attributes:
        _model : nn.FaceswapModel
         the generator of the faceswap-GAN model
    """

    def __init__(self, arch_config: dict, model_path: str = None):
        self._model = FaceswapModel(**arch_config)

        if model_path is not None and not self._model.weights_loaded:
            self._model.load_weights(model_path)

        self.input_image = None
        self.input_size = None
        self.img_bgr = None
        self.roi = None
        self.roi_size = None
        self.ae_input = None
        self.ae_output = None
        self.ae_output_masked = None
        self.ae_output_bgr = None
        self.ae_output_a = None
        self.result = None
        self.result_raw_rgb = None
        self.result_alpha = None
        pass

    def transform(self, input_image, direction, roi_coverage, color_correction, image_shape):
        self.check_generator_model(self.model)
        self.check_roi_coverage(input_image, roi_coverage)
        self.input_image = input_image

        # pre-process input image
        self._preprocess_inp_img(roi_coverage, image_shape)

        # model inference
        self._ae_forward_pass(direction)

        # post-process transformed roi image
        self._postprocess_roi_img(self.roi, color_correction)

        # merge transformed output back to input image
        self._merge_img_and_mask(self.roi, roi_coverage)

        return self.result, self.result_raw_rgb, self.result_alpha

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m: FaceswapModel):
        self._model = m
        pass

    @staticmethod
    def get_feather_edges_mask(image, roi_coverage):
        img_size = image.shape
        mask = np.zeros_like(image)
        roi_x, roi_y = (int(img_size[0] * (1 - roi_coverage)),
                        int(img_size[1] * (1 - roi_coverage)))
        mask[roi_x:-roi_x, roi_y:-roi_y, :] = 255
        mask = cv2.GaussianBlur(mask, (15, 15), 10)
        return mask

    @staticmethod
    def check_generator_model(model):
        if model is None:
            raise ValueError(f"Generator model has not been set.")
        pass

    @staticmethod
    def check_roi_coverage(inp_img, roi_coverage):
        input_size = inp_img.shape
        roi_x, roi_y = int(input_size[0] * (1 - roi_coverage)), int(input_size[1] * (1 - roi_coverage))
        if roi_x == 0 or roi_y == 0:
            raise ValueError("Error occurs when cropping roi image. \
            Consider increasing min_face_area or decreasing roi_coverage.")
        pass

    def _preprocess_inp_img(self, roi_coverage, image_shape):
        self.img_bgr = cv2.cvtColor(self.input_image, cv2.COLOR_RGB2BGR)
        self.input_size = self.img_bgr.shape
        roi_x, roi_y = int(self.input_size[0] * (1 - roi_coverage)), int(self.input_size[1] * (1 - roi_coverage))
        self.roi = self.img_bgr[roi_x:-roi_x, roi_y:-roi_y, :]  # BGR, [0, 255]
        self.roi_size = self.roi.shape
        self.ae_input = cv2.resize(self.roi, image_shape[:2]) / 255. * 2 - 1  # BGR, [-1, 1]
        pass

    def _ae_forward_pass(self, direction):
        ae_out = self.model.transform(self.ae_input, direction)
        self.ae_output = np.squeeze(np.array([ae_out]))
        pass

    def _postprocess_roi_img(self, roi, color_correction):
        ae_output_a = self.ae_output[:, :, 0] * 255
        ae_output_a = cv2.resize(ae_output_a, (self.roi_size[1], self.roi_size[0]))[..., np.newaxis]
        ae_output_bgr = np.clip((self.ae_output[:, :, 1:] + 1) * 255 / 2, 0, 255)
        ae_output_bgr = cv2.resize(ae_output_bgr, (self.roi_size[1], self.roi_size[0]))
        ae_output_masked = (ae_output_a / 255 * ae_output_bgr + (1 - ae_output_a / 255) * roi).astype('uint8')
        self.ae_output_a = ae_output_a
        if color_correction == ColorCorrectionType.ADAIN:
            self.ae_output_masked = adain(ae_output_masked, roi)
            self.ae_output_bgr = adain(ae_output_bgr, roi)
        elif color_correction == ColorCorrectionType.ADAIN_XYZ:
            self.ae_output_masked = adain(ae_output_masked, roi, color_space="XYZ")
            self.ae_output_bgr = adain(ae_output_bgr, roi, color_space="XYZ")
        elif color_correction == ColorCorrectionType.HISTMATCH:
            self.ae_output_masked = color_hist_match(ae_output_masked, roi)
            self.ae_output_bgr = color_hist_match(ae_output_bgr, roi)
        else:
            self.ae_output_masked = ae_output_masked
            self.ae_output_bgr = ae_output_bgr
            pass
        pass

    def _merge_img_and_mask(self, roi, roi_coverage):
        blend_mask = self.get_feather_edges_mask(roi, roi_coverage)
        blended_img = (blend_mask / 255 * self.ae_output_masked) + (1 - blend_mask / 255) * roi
        roi_x, roi_y = int(self.input_size[0] * (1 - roi_coverage)), int(self.input_size[1] * (1 - roi_coverage))
        self.result = self.img_bgr.copy()
        self.result[roi_x:-roi_x, roi_y:-roi_y, :] = blended_img
        self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)
        self.result_raw_rgb = self.img_bgr.copy()
        self.result_raw_rgb[roi_x:-roi_x, roi_y:-roi_y, :] = self.ae_output_bgr
        self.result_raw_rgb = cv2.cvtColor(self.result_raw_rgb, cv2.COLOR_BGR2RGB)
        self.result_alpha = np.zeros_like(self.img_bgr)
        self.result_alpha[roi_x:-roi_x, roi_y:-roi_y, :] = (blend_mask / 255) * self.ae_output_a
        pass

    pass
