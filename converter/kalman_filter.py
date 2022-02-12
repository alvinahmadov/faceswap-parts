import cv2
import numpy as np


class KalmanFilter:
    def __init__(self, noise_coefficient):
        self.noise_coef = noise_coefficient
        self.kalman_filter = self.init_kalman_filter(noise_coefficient)
        pass

    def correct(self, xy):
        return self.kalman_filter.correct(xy)

    def predict(self):
        return self.kalman_filter.predict()

    @staticmethod
    def init_kalman_filter(noise_coefficient):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        kf.processNoiseCov = noise_coefficient * np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        return kf

    pass
