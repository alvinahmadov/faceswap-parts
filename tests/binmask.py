from glob import glob
from pathlib import PurePath, Path

import cv2
import numpy as np
from face_alignment import FaceAlignment
from matplotlib import pyplot as plt

from converter.config import TransformDirection
from converter.segmentation import SegmentationType

DIR_TRAIN = "/home/alvin/faceswap_train"
DIR_FACE_A = f"{DIR_TRAIN}/faceA"
DIR_FACE_B = f"{DIR_TRAIN}/faceB"
DIR_FACE_A_RGB = f"{DIR_FACE_A}/rgb"
DIR_FACE_B_RGB = f"{DIR_FACE_B}/rgb"
DIR_BM_FACE_A = f"{DIR_FACE_A}/binary_masks"
DIR_BM_FACE_B = f"{DIR_FACE_B}/binary_masks"
FNS_FACE_A = glob(f"{DIR_FACE_A_RGB}/*.*")
FNS_FACE_B = glob(f"{DIR_FACE_B_RGB}/*.*")

EYE_RIGHT = (36, 42)
EYE_LEFT = (42, 48)
NOSE = (27, 35)
MOUTH = (48, 68)

Path(DIR_BM_FACE_A).mkdir(parents=True, exist_ok=True)
Path(DIR_BM_FACE_B).mkdir(parents=True, exist_ok=True)

fns_face_not_detected = []


def draw_mask_contours(predictions, mask, pnts):
    pnts = [(predictions[i, 0], predictions[i, 1]) for i in range(pnts[0], pnts[1])]
    hull = cv2.convexHull(np.array(pnts)).astype(np.int32)
    return cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)


def face_segmentation(mask, preds, option: SegmentationType, shape: tuple = None):
    if shape:
        mask = cv2.resize(mask, shape)
        pass

    if option == SegmentationType.EYES_ONLY:
        mask = draw_mask_contours(preds, mask, EYE_RIGHT)
        mask = draw_mask_contours(preds, mask, EYE_LEFT)
        pass
    elif option == SegmentationType.NOSE_ONLY:
        mask = draw_mask_contours(preds, mask, NOSE)
        pass
    elif option == SegmentationType.MOUTH_ONLY:
        mask = draw_mask_contours(preds, mask, MOUTH)
        pass
    elif option == SegmentationType.ALL:
        mask = draw_mask_contours(preds, mask, EYE_RIGHT)  # Draw right eye binary mask
        mask = draw_mask_contours(preds, mask, EYE_LEFT)  # Draw left eye binary mask
        mask = draw_mask_contours(preds, mask, NOSE)  # Draw nose binary mask
        mask = draw_mask_contours(preds, mask, MOUTH)  # Draw mouth binary mask
        pass
    mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    return mask


def get_info():
    num_faceA = len(glob(f"{DIR_FACE_A_RGB}/*.*"))
    num_faceB = len(glob(f"{DIR_FACE_B_RGB}/*.*"))

    print(f"Nuber of processed images: {num_faceA + num_faceB}")
    print(f"Number of image(s) with no face detected: {len(fns_face_not_detected)}")
    pass


def seg(option, direction):
    if direction == TransformDirection.AtoB:
        pass
    elif direction == TransformDirection.BtoA:
        pass
    else:
        raise

    fa = FaceAlignment(1, device='cpu', flip_input=False)
    for idx, fns in enumerate([FNS_FACE_A, FNS_FACE_B]):
        save_path = DIR_BM_FACE_A if direction == TransformDirection.AtoB else DIR_BM_FACE_B
        save_path_comb = f"{save_path}/comb"
        save_path_mask = f"{save_path}/mask"

        if not Path(save_path_comb).exists():
            Path(save_path_comb).mkdir(parents=True, exist_ok=True)

        if not Path(save_path_mask).exists():
            Path(save_path_mask).mkdir(parents=True, exist_ok=True)

        # create binary mask for each training image
        for fn in fns:
            raw_fn = PurePath(fn).parts[-1]

            image = cv2.imread(fn)
            image = cv2.resize(image, (256, 256))
            preds = fa.get_landmarks(image)
            mask = np.zeros_like(image)

            if preds is not None:
                preds = preds[0]
                mask = face_segmentation(mask, preds, option, shape=(256, 256))
                pass
            else:
                print(f"No faces were detected in image '{fn}'")
                fns_face_not_detected.append(fn)
                pass

            bm_comb = (mask / 255 * image).astype(np.uint8)

            plt.imsave(fname=f"{save_path_comb}/{raw_fn}.png", arr=bm_comb, format="png")
            plt.imsave(fname=f"{save_path_mask}/{raw_fn}.png", arr=mask, format="png")
            pass
        pass
    pass


def show_random():
    face = np.random.choice(["A", "B"])

    dir_face = DIR_FACE_A_RGB if face == "A" else DIR_FACE_B_RGB
    fns_face = FNS_FACE_A if face == "A" else FNS_FACE_B
    num_face = len(glob(dir_face + "/*.*"))
    rand_idx = np.random.randint(num_face)
    rand_fn = fns_face[rand_idx]
    raw_fn = PurePath(rand_fn).parts[-1]
    mask_fn = f"{DIR_BM_FACE_A}/{raw_fn}" if face == "A" else f"{DIR_BM_FACE_B}/{raw_fn}"
    # resize_shape = (256, 256)

    im = cv2.imread(rand_fn)
    mask = cv2.imread(mask_fn)
    bm_comb = (mask / 255 * im).astype(np.uint8)

    # im = cv2.resize(im, resize_shape)
    # mask = cv2.resize(mask, resize_shape)

    if rand_fn in fns_face_not_detected:
        print("========== На этом изображении не было обнаружено никаких лиц! ==========")

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.grid('off')
    plt.imshow(im)
    plt.subplot(1, 3, 2)
    plt.grid('off')
    plt.imshow(mask)
    plt.subplot(1, 3, 3)
    plt.grid('off')
    plt.imshow(bm_comb)
    pass


if __name__ == '__main__':
    opt = SegmentationType.ALL
    seg(opt, TransformDirection.AtoB)
    pass
