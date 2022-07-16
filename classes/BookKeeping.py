import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2

# Class BookKeeping holds three lists that only consists of processed data
# The list entries are objects from the classes Observation, View and Point3D
class BookKeeper:
    def __init__(self):
        self.observations = []  #T_obs i IREG
        self.views = []         #T_views
        self.points3d = []      #T_points
    
    def add_observation(self, observation):
        self.observations.append(observation)

    def add_view(self, view):
        if view.R is None or view.t is None:
            raise ValueError("Pose needs to be defined for the view before added to Q.")

        self.views.append(view)

    def add_point3d(self, point3d):
        if point3d.coordinates.shape[0] != 3:
            raise ValueError('Points array needs to be N x 3.')

        self.points3d.append(point3d)

    def bookkeep_observation(self, view, point_2d, point3d):
        """
        Parameters
        ----------------
        view : View-object
        point_2d : (2, 1) or (2, ) array
                    The coordinates of the image point
        point3d : Point3D-object
        """
        observation = Observation(point_2d, view, point3d)
        self.add_observation(observation)
        view.add_observation(observation)
        point3d.add_observation(observation)

class View:
    def __init__(self, image, points_2d, proj_mtx):
        self.image = image
        self.points_2d = points_2d
        self.proj_mtx = proj_mtx
        self.R = None
        self.t = None
        self.observations = []

    def set_pose(self, R, t):
        self.R = R
        self.t = t

    def get_external_mtx(self):
        return np.concatenate((self.R, self.t), axis=1)

    def add_observation(self, observation):
        self.observations.append(observation)

    def get_rot_as_quat(self):
        r = Rot.from_matrix(self.R)
        return r.as_quat()

    def get_rot_as_rodrigues(self):
        r, _ = cv2.Rodrigues(self.R)
        return r

class Point3D:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.observations = []

    def add_observation(self, observation):
        self.observations.append(observation)

class Observation:
    def __init__(self, point_2d, view, point_3d):
        self.point_2d = point_2d
        self.view = view
        self.point_3d = point_3d









