from classes.BookKeeping import *
from loaders.image_loader import image_loader
from scipy import io
import numpy as np
import cv2

def decomposeP(P):
    # This is an alternative way of decomposing P. Gives same result as cv2.decomposeProjectionMatrix
    M = P[0:3,0:3]
    Q = np.eye(3)[::-1]
    P_b = Q @ M @ M.T @ Q
    K_h = Q @ np.linalg.cholesky(P_b) @ Q
    K = K_h / K_h[2,2]
    A = np.linalg.inv(K) @ M
    l = (1/np.linalg.det(A)) ** (1/3)
    R = l * A
    t = l * np.linalg.inv(K) @ P[0:3,3]
    return K, R, t

def setup_views():
    views = []

    images = image_loader("dataset/images/*.ppm")

    dino_gt = io.loadmat('dataset/BADino2.mat')
    all_points_2d = dino_gt['newPoints2D']
    #dino_test = io.loadmat('dataset/dino_Ps.mat')
    projection_matrices = dino_gt['newPs']

    # All views should have the same K. Just decompose it from first C.
    first_proj_mtx = projection_matrices[:, 0][0]
    (K, R, t) = cv2.decomposeProjectionMatrix(first_proj_mtx)[0:3]
    K = np.matrix(K)
    #(K1, R1, t1) = decomposeP(first_proj_mtx)
    # Important: Divide t by t[3] to make it homogeneous. This ‘t‘ is not the one in P = [R|t]. It is the one in P = [R | R(-t)]

    for i, image in enumerate(images):
        points_2d = all_points_2d[:, i][0]
        proj_mtx = projection_matrices[:, i][0]

        views.append(View(image, points_2d, proj_mtx))

    return (views, K)