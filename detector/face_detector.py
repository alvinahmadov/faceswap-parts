import os

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K

from detector import mtcnn


class MTCNNFaceDetector:
    def __init__(self, session=None, model_path=None):
        """
        Loads the MTCNN network and performs face detection.

        Parameters
        ----------
        session :
         tensorflow session
        model_path : str
         path to the trained models for mtcnn
        """
        self.pnet: mtcnn.Network
        self.rnet: mtcnn.Network
        self.onet: mtcnn.Network
        self.session = session if session is not None else K.get_session()
        self.model_path = model_path
        self._create()
        pass

    def detect_face(self, image, minsize=20, threshold=0.7, factor=0.709,
                    use_auto_downscaling=True, min_face_area=25 * 25):
        """
        Detects faces in an image, and returns bounding boxes and points for them.

        Parameters
        ------
        image : np.ndarray
         input image
        minsize : int
         minimum faces' size
        threshold :
         threshold=[th1, th2, th3], th1-3 are three steps's threshold
        factor : float
         the factor used to create a scaling pyramid of face sizes to detect in the image.
        use_auto_downscaling : bool
         downscale image
        min_face_area : float
         minimal area of face to be detected
        """
        scale_factor = None
        if use_auto_downscaling:
            image, scale_factor = self._auto_downscale(image)
            pass

        faces, pnts = mtcnn.detect_face(
            image, minsize,
            self.pnet, self.rnet, self.onet,
            [0.6, 0.7, threshold],
            factor
        )
        faces = self.process_mtcnn_bbox(faces, image.shape)
        faces, pnts = self.remove_small_faces(faces, pnts, min_face_area)

        if use_auto_downscaling:
            faces = self.calibrate_coord(faces, scale_factor)
            pnts = self.calibrate_landmarks(pnts, scale_factor)
            pass
        return faces, pnts

    @staticmethod
    def process_mtcnn_bbox(bboxes, im_shape):
        """
        Process the bbox coordinates to a square bbox with ordering (x0, y1, x1, y0)

        Parameters
        ------
        bboxes :
         face bounding boxes
        im_shape :
         shape of image (resolution)
        """
        for i, bbox in enumerate(bboxes):
            y0, x0, y1, x1 = bboxes[i, 0:4]
            w = int(y1 - y0)
            h = int(x1 - x0)
            length = (w + h) / 2
            center = (int((x1 + x0) / 2), int((y1 + y0) / 2))
            new_x0 = np.max([0, (center[0] - length // 2)])
            new_x1 = np.min([im_shape[0], (center[0] + length // 2)])
            new_y0 = np.max([0, (center[1] - length // 2)])
            new_y1 = np.min([im_shape[1], (center[1] + length // 2)])
            bboxes[i, 0:4] = new_x0, new_y1, new_x1, new_y0
            pass
        return bboxes

    @staticmethod
    def calibrate_coord(faces, scale_factor):
        for i, (x0, y1, x1, y0, _) in enumerate(faces):
            faces[i] = (x0 * scale_factor, y1 * scale_factor,
                        x1 * scale_factor, y0 * scale_factor, _)
            pass
        return faces

    @staticmethod
    def calibrate_landmarks(pnts: np.ndarray, scale_factor: float):
        return np.array([xy * scale_factor for xy in pnts])

    @staticmethod
    def remove_small_faces(faces, pnts, min_area=25. * 25.):
        """
        Filter faces with unsufficient face area

        Parameters
        ----------
        faces : ndarray
         All detected faces
        pnts : ndarray
         All detected face points
        min_area : float
         minimal face area to pass faces and points

        Returns
        -------
        Tuple of filtered faces and face points
        """
        def compute_area(face_coord):
            x0, y1, x1, y0, _ = face_coord
            area = np.abs((x1 - x0) * (y1 - y0))
            return area

        new_faces = []
        new_pnts = []
        # faces has shape (num_faces, coord), and pnts has shape (coord, num_faces)
        for face, pnt in zip(faces, pnts.transpose()):
            if compute_area(face) >= min_area:
                new_faces.append(face)
                new_pnts.append(pnt)
                pass
            pass
        new_faces = np.array(new_faces)
        new_pnts = np.array(new_pnts).transpose()
        return new_faces, new_pnts

    @staticmethod
    def is_higher_than_480p(x):
        return (x.shape[0] * x.shape[1]) >= (858 * 480)

    @staticmethod
    def is_higher_than_720p(x):
        return (x.shape[0] * x.shape[1]) >= (1280 * 720)

    @staticmethod
    def is_higher_than_1080p(x):
        return (x.shape[0] * x.shape[1]) >= (1920 * 1080)

    def _create(self):
        """
        Creates MTCNN Network
        """
        if not self.model_path:
            self.model_path, _ = os.path.split(os.path.realpath(__file__))
            pass

        with tf.variable_scope('pnet'):
            data = tf.placeholder(shape=(None, None, None, 3), dtype=tf.float32, name='input')
            pnet = mtcnn.PNet({'data': data})
            pnet.load(os.path.join(self.model_path, 'det1.npy'), self.session)
            pass
        with tf.variable_scope('rnet'):
            data = tf.placeholder(shape=(None, 24, 24, 3), dtype=tf.float32, name='input')
            rnet = mtcnn.RNet({'data': data})
            rnet.load(os.path.join(self.model_path, 'det2.npy'), self.session)
            pass
        with tf.variable_scope('onet'):
            data = tf.placeholder(shape=(None, 48, 48, 3), dtype=tf.float32, name='input')
            onet = mtcnn.ONet({'data': data})
            onet.load(os.path.join(self.model_path, 'det3.npy'), self.session)
            pass

        self.pnet = K.function(
            [pnet.layers['data'], ],
            [pnet.layers['conv4-2'], pnet.layers['prob1']]
        )
        self.rnet = K.function(
            [rnet.layers['data'], ],
            [rnet.layers['conv5-2'], rnet.layers['prob1']]
        )
        self.onet = K.function(
            [onet.layers['data'], ],
            [onet.layers['conv6-2'], onet.layers['conv6-3'], onet.layers['prob1']]
        )
        pass

    def _auto_downscale(self, image):
        if self.is_higher_than_1080p(image):
            scale_factor = 4
            resized_image = cv2.resize(image,
                                       (image.shape[1] // scale_factor,
                                        image.shape[0] // scale_factor))
            pass
        elif self.is_higher_than_720p(image):
            scale_factor = 3
            resized_image = cv2.resize(image,
                                       (image.shape[1] // scale_factor,
                                        image.shape[0] // scale_factor))
            pass
        elif self.is_higher_than_480p(image):
            scale_factor = 2
            resized_image = cv2.resize(image,
                                       (image.shape[1] // scale_factor,
                                        image.shape[0] // scale_factor))
            pass
        else:
            scale_factor = 1
            resized_image = image.copy()
            pass
        return resized_image, scale_factor

    pass
