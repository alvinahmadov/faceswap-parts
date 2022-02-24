import cv2
import face_alignment as fa
import numpy as np

from .config import SegmentationType


class FaceSegmentation:
    """
    Extracts face segments from face image to swap with original face
    """

    EYEBROW_RIGHT = (17, 22)
    EYEBROW_LEFT = (22, 27)
    NOSE = (27, 35)
    EYE_RIGHT = (36, 42)
    EYE_LEFT = (42, 48)
    MOUTH = (48, 68)

    BLUR_KERNEL = (1, 1)
    DILATE_KERNEL = (9, 9)

    def __init__(self, device='cpu'):
        """
        Makes face parts segmentation in image
        Parameters
        ----------
        device : str
         Device to use.
         Available: cuda, cpu
        """
        self.face_alignment = fa.FaceAlignment(1, device=device, flip_input=False)
        pass

    def get_segments(self, image_a: np.ndarray, image_b: np.ndarray,
                     seg_type: SegmentationType = SegmentationType.ALL):
        """
        Do segmentation
        Parameters
        ----------
        image_a : ndarray
         Image to feed for face landmarks detection
        image_b : ndarray
         Image to apply face segmentation from image_a
        seg_type : SegmentationType
         Face part option to get_segments

        Returns
        -------
        Image with specific face part

        """
        # get source face landmark predictions
        predictions = self.face_alignment.get_landmarks(image_a)

        if predictions is not None:
            preds = predictions[0]

            # extract mask
            mask = self._get_face_mask(image_b, preds, seg_type)

            # merge target image and source mask
            result = (mask / 255 * image_b).astype(np.uint8)

            # find zeros in result then replace with values from source image
            idx = np.where(result <= 10)
            result[idx] = image_a[idx]
            pass
        else:
            print(f"No faces were detected in image")
            result = image_b
            pass

        return result

    def _get_face_mask(self, image, preds, option):
        """
        Choose face part and draw it with convex hull its contours white
        Parameters
        ----------
        image : ndarray
        preds :
         List of predicted face landmarks to extract
        option : SegmentationType
         Part or parts of face to extract

        Returns
        -------

        """
        mask = np.zeros_like(image)
        if option == SegmentationType.EYES_ONLY:
            mask = self._draw_mask_contours(mask, preds, self.EYE_RIGHT)
            mask = self._draw_mask_contours(mask, preds, self.EYE_LEFT)
            pass
        if option == SegmentationType.EYEBROWS_ONLY:
            mask = self._draw_mask_contours(mask, preds, self.EYEBROW_RIGHT)
            mask = self._draw_mask_contours(mask, preds, self.EYEBROW_LEFT)
            pass
        elif option == SegmentationType.NOSE_ONLY:
            mask = self._draw_mask_contours(mask, preds, self.NOSE)
            pass
        elif option == SegmentationType.MOUTH_ONLY:
            mask = self._draw_mask_contours(mask, preds, self.MOUTH)
            pass
        elif option == SegmentationType.EYES_AND_EYEBROWS:
            mask = self._draw_mask_contours(mask, preds, self.EYE_RIGHT)
            mask = self._draw_mask_contours(mask, preds, self.EYE_LEFT)
            mask = self._draw_mask_contours(mask, preds, self.EYEBROW_RIGHT)
            mask = self._draw_mask_contours(mask, preds, self.EYEBROW_LEFT)
            pass
        elif option == SegmentationType.EYES_AND_NOSE:
            mask = self._draw_mask_contours(mask, preds, self.EYE_RIGHT)
            mask = self._draw_mask_contours(mask, preds, self.EYE_LEFT)
            mask = self._draw_mask_contours(mask, preds, self.NOSE)
            pass
        elif option == SegmentationType.EYES_AND_MOUTH:
            mask = self._draw_mask_contours(mask, preds, self.EYE_RIGHT)
            mask = self._draw_mask_contours(mask, preds, self.EYE_LEFT)
            mask = self._draw_mask_contours(mask, preds, self.MOUTH)
            pass
        elif option == SegmentationType.EYEBROWS_AND_NOSE:
            mask = self._draw_mask_contours(mask, preds, self.EYEBROW_RIGHT)
            mask = self._draw_mask_contours(mask, preds, self.EYEBROW_LEFT)
            mask = self._draw_mask_contours(mask, preds, self.NOSE)
            pass
        elif option == SegmentationType.MOUTH_AND_NOSE:
            mask = self._draw_mask_contours(mask, preds, self.NOSE)
            mask = self._draw_mask_contours(mask, preds, self.MOUTH)
            pass
        elif option == SegmentationType.ALL:
            mask = self._draw_mask_contours(mask, preds, self.EYE_RIGHT)
            mask = self._draw_mask_contours(mask, preds, self.EYE_LEFT)
            mask = self._draw_mask_contours(mask, preds, self.EYEBROW_RIGHT)
            mask = self._draw_mask_contours(mask, preds, self.EYEBROW_LEFT)
            mask = self._draw_mask_contours(mask, preds, self.NOSE)
            mask = self._draw_mask_contours(mask, preds, self.MOUTH)
            pass
        mask = cv2.dilate(mask, np.ones(self.DILATE_KERNEL, np.uint8), iterations=2)
        return cv2.GaussianBlur(mask, self.BLUR_KERNEL, 0)

    @staticmethod
    def _draw_mask_contours(mask, preds, pnts):
        """

        Parameters
        ----------
        mask
        pnts

        Returns
        -------

        """
        pnts = [(preds[i, 0], preds[i, 1]) for i in range(pnts[0], pnts[1])]
        hull = cv2.convexHull(np.array(pnts)).astype(np.int32)
        return cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

    pass
