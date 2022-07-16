import cv2
import numpy as np
from ext import lab3
from classes.BookKeeping import *
from init.get_correspondences import get_correspondences
from ext import lab3
import matplotlib.pyplot as plt
from _tests_.test_pose_extraction import test_pose_extraction

def get_3d2dcorr(prev_view, new_view):
    prev_corr, new_corr = get_correspondences(prev_view, new_view)
    prev_obs = prev_view.observations

    corr_3d = []
    corr_2d = []
    not_triangulated_prev = []
    not_triangulated_new = []
    for i in range(new_corr.shape[1]):
        corr = get_corr3d(prev_obs, prev_corr[:, i])
        if corr is not None:
            corr_2d.append(new_corr[:, i])
            corr_3d.append(corr)
        else:
            not_triangulated_prev.append(prev_corr[:, i])
            not_triangulated_new.append(new_corr[:, i])

    return corr_2d, corr_3d, not_triangulated_prev, not_triangulated_new

# checks if a point is an observation, i.e exists in the observation list, and returns the observation.
def get_corr3d(obs_list, point_2d):
    for o in obs_list:
        if np.array_equal(point_2d, o.point_2d):
            return o.point_3d
    return None

def add_new_3dpoints(prev_view, new_view, not_triangulated_prev, not_triangulated_new, K, Keeper):
    #K[0,1] = 0
    C1 = np.array(K @ prev_view.get_external_mtx())
    C2 = np.array(K @ new_view.get_external_mtx())
    # TODO: compute E
    for i in range(len(not_triangulated_new)):
        # TODO: check so the points fulfill epipolar constraint relative to E. If they do:
        point_3d_coords = lab3.triangulate_optimal(C1, C2, not_triangulated_prev[i], not_triangulated_new[i])
        point3d = Point3D(point_3d_coords)
        Keeper.add_point3d(point3d)
        Keeper.bookkeep_observation(prev_view, not_triangulated_prev[i], point3d)
        Keeper.bookkeep_observation(new_view, not_triangulated_new[i], point3d)
        #test_pose_extraction(prev_view, not_triangulated_prev, K, not_triangulated_prev[i], not_triangulated_new[i], point3d.coordinates)

def add_view(prev_view, new_view, K, Keeper):
    corr2d, corr3d, not_triangulated_prev, not_triangulated_new = get_3d2dcorr(prev_view, new_view)

    # make the lists into arrays of right dimensions for PnP
    corr2d_array = np.zeros((len(corr2d), 2))
    corr3d_array = np.zeros((len(corr2d), 3))
    for i in range(len(corr2d)):
        corr2d_array[i, :] = corr2d[i]
        corr3d_array[i, :] = corr3d[i].coordinates
    corr2d_array_shaped = np.reshape(corr2d_array, (len(corr2d), 1, 2)) #dim Nx1x2
    corr3d_array_shaped = np.reshape(corr3d_array, (len(corr2d), 1, 3)) #dim Nx1x3

    # PNP
    _, Rvec, t, _ = cv2.solvePnPRansac(corr3d_array_shaped, corr2d_array_shaped, K, None, flags=cv2.SOLVEPNP_EPNP)
    R, _ = cv2.Rodrigues(Rvec)

    new_view.set_pose(R, t)
    Keeper.add_view(new_view)

    pt_3d = np.append(corr3d[0].coordinates, 1)

    # TODO: build concensus set. If points belong to concensus: add to T_obs:
    for i in range(corr2d_array.shape[0]):
        Keeper.bookkeep_observation(new_view, corr2d_array.T[:,i], corr3d[i])

    # add 3dpoints for the corresponding points that has not yet been triangulated
    add_new_3dpoints(prev_view, new_view, not_triangulated_prev, not_triangulated_new, K, Keeper)
    

