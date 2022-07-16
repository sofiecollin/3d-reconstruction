import numpy as np

class P:
    def __init__(self):
        self.points_3d = None

    def append_points_3d(self, points):
        if points.shape[1] != 3:
            raise ValueError('Points array needs to be N x 3.')

        if self.points_3d is None:
            self.points_3d = points
        else:
            self.points_3d = np.concatenate((self.points_3d, points), axis=0)