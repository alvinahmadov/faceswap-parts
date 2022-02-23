import cv2
import numpy as np

from umeyama import umeyama

AVERAGE_LANDMARKS = [
    (0.31339227236234224, 0.3259269274198092),
    (0.31075140146108776, 0.7228453709528997),
    (0.5523683107816256, 0.5187296867370605),
    (0.7752419985257663, 0.37262483743520886),
    (0.7759613623985877, 0.6772957581740159)
]


def get_src_landmarks(x0, y0, pnts, face_idx=0):
    """
    Parameters
    ------
    x0 : int
     X start coordinate of bounding box
    y0 : int
     Y start coordinate of bounding box
    pnts :
     Landmarks predicted by MTCNN
    face_idx : int
    """
    src_landmarks = [(int(pnts[i + 5][face_idx] - x0),
                      int(pnts[i][face_idx] - y0)) for i in range(5)]
    return src_landmarks


def get_tar_landmarks(img: np.ndarray):
    """
    Parameters
    ------
    img : np.ndarray
     Detected face image
    """

    img_sz = img.shape
    tar_landmarks = [
        (int(xy[0] * img_sz[0]), int(xy[1] * img_sz[1]))
        for xy in AVERAGE_LANDMARKS
    ]
    return tar_landmarks


# noinspection PyPep8Naming
def landmarks_match_mtcnn(src_image, src_landmarks, tar_landmarks):
    """
    src/dst landmarks coordinates should be (y, x)
    """
    src_size = src_image.shape
    src_tmp = [(int(xy[1]), int(xy[0])) for xy in src_landmarks]
    dst_tmp = [(int(xy[1]), int(xy[0])) for xy in tar_landmarks]
    M = umeyama(np.array(src_tmp), np.array(dst_tmp), True)[0:2]
    result = cv2.warpAffine(src_image, M, (src_size[1], src_size[0]), borderMode=cv2.BORDER_REPLICATE)
    return result
