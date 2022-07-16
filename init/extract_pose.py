import numpy as np
from ext import lab3
import cv2

def ssvd(E):
    # From Mathematical Toolbox for Studies in Visual Computation at Linköping University, version 0.40
    # by Klas Nordberg

    (U, S, Vt) = np.linalg.svd(E)
    Up = U.copy()
    Up[:, -1] = np.linalg.det(U) * Up[:, -1]
    V = Vt.T
    Vp = V.copy()
    Vp[:, -1] = np.linalg.det(V) * Vp[:, -1]

    return (Up, None, Vp.T)

def is_front_of_cameras(pt1, pt2, R, t):
    C1 = np.eye(3, 4)
    C2 = np.concatenate((R, t), axis=1)

    X = lab3.triangulate_linear(np.asarray(C1), np.asarray(C2), np.asarray(pt1), np.asarray(pt2))
    X2 = R @ X[:, None] + t

    if X[2] > 0 and X2[2, 0] > 0:
        return True
    else: return False

def find_valid_pose(pt1, pt2, R1, R2, t1, t2):
    # Should only go into one of these cases:
    if is_front_of_cameras(pt1, pt2, R1, t1):
        return (R1, t1)
    if is_front_of_cameras(pt1, pt2, R2, t1):
        return (R2, t1)
    if is_front_of_cameras(pt1, pt2, R1, t2):
        return (R1, t2)
    if is_front_of_cameras(pt1, pt2, R2, t2):
        return (R2, t2)

def extract_pose(E, K, pt1, pt2):
    pt1_norm = np.append(pt1, 1)
    pt1_norm = np.linalg.inv(K) @ pt1_norm[:, None] # Convert to C-normalized coordinates
    pt2_norm = np.append(pt2, 1)
    pt2_norm = np.linalg.inv(K) @ pt2_norm[:, None]

    #E = np.matrix([[0.0101, 0.8140, 0.5804], [-0.7922, 0.0092, 0.0225], [-0.6099, -0.0141, 0.0022]])
    (U, _, Vt) = ssvd(E)

    V = Vt.T

    W = np.matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    R1 = V @ W @ U.T
    R2 = V @ W.T @ U.T

    t1 = V[:, -1] / np.linalg.norm(V[: ,-1])
    t2 = -t1

    (R, t) = find_valid_pose(pt1_norm, pt2_norm, R1, R2, t1, t2)

    return R, t