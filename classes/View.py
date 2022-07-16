import numpy as np
from scipy.spatial.transform import Rotation
import cv2

class View:
    def __init__(self, image, points_2d, proj_mtx):
        self.image = image
        self.points_2d = points_2d
        self.proj_mtx = proj_mtx
        self.R = None
        self.t = None

    def set_pose(self, R, t):
        self.R = R
        self.t = t

    def get_external_mtx(self):
        return np.concatenate((self.R, self.t), axis=1)

    # Check if this produces correct 3 value rotation
    def get_external_mtx_rodrigues(self):
        R, _ = cv2.Rodrigues(self.R)
        return np.concatenate((R, self.t), axis=1)

