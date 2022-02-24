import cv2
import numpy as np


def draw_landmarks(image, landmarks: list, filename: str):
    color = (0, 255, 0)
    for i, lm in enumerate(landmarks):
        _image = cv2.circle(image, (lm[1], lm[0]), 2, color, -1)
        _image = cv2.putText(_image, str(i), (lm[1], lm[0]), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        cv2.imwrite(filename, _image)
    pass


def fillzeros(fname, maxlen=3, ext=".png"):
    def _fill(ln, ch='0'):
        if len(ln) < maxlen:
            c = maxlen - len(ln)
            ln = ch * c + ln
            return ln
        else:
            return fname
        pass

    if isinstance(fname, str):
        return _fill(fname)
        pass
    elif isinstance(fname, int):
        return _fill(str(fname))
        pass
    else:
        return fname
    pass


def get_init_mask_map(image):
    return np.zeros_like(image)


def get_init_comb_img(input_img) -> np.ndarray:
    """
    Get inital combination image

    Parameters
    ----------
    input_img : ndarray
     Input image

    """
    comb_img = np.zeros([input_img.shape[0], input_img.shape[1] * 2, input_img.shape[2]])
    comb_img[:, :input_img.shape[1], :] = input_img
    comb_img[:, input_img.shape[1]:, :] = input_img
    return comb_img


def get_init_triple_img(input_img, no_face=False):
    if no_face:
        triple_img = np.zeros([input_img.shape[0], input_img.shape[1] * 3, input_img.shape[2]])
        triple_img[:, :input_img.shape[1], :] = input_img
        triple_img[:, input_img.shape[1]:input_img.shape[1] * 2, :] = input_img
        triple_img[:, input_img.shape[1] * 2:, :] = (input_img * .15).astype('uint8')
        return triple_img
    else:
        triple_img = np.zeros([input_img.shape[0], input_img.shape[1] * 3, input_img.shape[2]])
        return triple_img
    pass


def get_mask(roi_image, h, w):
    mask = np.zeros_like(roi_image)
    mask[h // 15:-h // 15, w // 15:-w // 15, :] = 255
    mask = cv2.GaussianBlur(mask, (15, 15), 10)
    return mask
