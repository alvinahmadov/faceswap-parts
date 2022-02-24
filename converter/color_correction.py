import enum

import cv2
import numpy as np


class ColorCorrectionType(enum.IntEnum):
    NONE = 0
    ADAIN = 1
    ADAIN_XYZ = 2
    HISTMATCH = 3
    pass


"""Color correction functions"""


def hist_match(source: np.ndarray, template):
    """
    Histogram matching of two images

    Parameters
    ----------
    source : np.ndarray
     Source image

    template : np.ndarray
     Template

    References
    -------
    Code borrow from:
    https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    """

    old_shape = source.shape
    source = source.ravel()
    template = template.ravel()
    s_values, bin_idx, s_counts = np.unique(
        source, return_inverse=True, return_counts=True
    )
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(old_shape)


def color_hist_match(img_a, img_b, color_space="RGB"):
    if color_space.lower() != "rgb":
        img_a = trans_color_space(img_a, color_space)
        img_b = trans_color_space(img_b, color_space)
        pass

    matched_r = hist_match(img_a[:, :, 0], img_b[:, :, 0])
    matched_g = hist_match(img_a[:, :, 1], img_b[:, :, 1])
    matched_b = hist_match(img_a[:, :, 2], img_b[:, :, 2])
    matched = np.stack((matched_r, matched_g, matched_b), axis=2).astype(np.float32)
    matched = np.clip(matched, 0, 255)
    return matched


def adain(img_a, img_b, eps=1e-7, color_space="RGB"):
    # https://github.com/ftokarev/tf-adain/blob/master/adain/norm.py
    if color_space.lower() != "rgb":
        img_a = trans_color_space(img_a, color_space)
        img_b = trans_color_space(img_b, color_space)
        pass

    mt = np.mean(img_b, axis=(0, 1))
    st = np.std(img_b, axis=(0, 1))
    ms = np.mean(img_a, axis=(0, 1))
    ss = np.std(img_a, axis=(0, 1))
    if ss.any() <= eps:
        return img_a
    result = st * (img_a.astype(np.float32) - ms) / (ss + eps) + mt
    result = np.clip(result, 0, 255)

    if color_space.lower() != "rgb":
        result = trans_color_space(result.astype(np.uint8), color_space, rev=True)
    return result


def trans_color_space(im, color_space, rev=False):
    rev_clr_spc = 0
    clr_spc = 0
    if color_space.lower() == "lab":
        clr_spc = cv2.COLOR_BGR2Lab
        rev_clr_spc = cv2.COLOR_Lab2BGR
    elif color_space.lower() == "ycbcr":
        clr_spc = cv2.COLOR_BGR2YCR_CB
        rev_clr_spc = cv2.COLOR_YCR_CB2BGR
    elif color_space.lower() == "xyz":
        clr_spc = cv2.COLOR_BGR2XYZ
        rev_clr_spc = cv2.COLOR_XYZ2BGR
    elif color_space.lower() == "luv":
        clr_spc = cv2.COLOR_BGR2Luv
        rev_clr_spc = cv2.COLOR_Luv2BGR
    elif color_space.lower() == "rgb":
        pass
    else:
        raise NotImplementedError()

    if color_space.lower() != "rgb":
        trans_clr_spc = rev_clr_spc if rev else clr_spc
        im = cv2.cvtColor(im, trans_clr_spc)
    return im
