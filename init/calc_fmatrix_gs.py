from ext import lab3
import numpy as np
from scipy.optimize import least_squares


def calc_fmatrix_gs(F, inliers1, inliers2):
    (C1, C2) = lab3.fmatrix_cameras(F)
    tri_points = []

    for i in range(inliers1.shape[1]):
        p = lab3.triangulate_optimal(C1, C2, inliers1[:, i], inliers2[:, i])
        tri_points.append(p)

    tri_points = np.asarray(tri_points)

    param1 = np.hstack((C1.ravel(), tri_points.ravel()))

    def cost(x):
        return lab3.fmatrix_residuals_gs(x, inliers1, inliers2)

    D2 = least_squares(cost, param1)

    C1_refined = np.reshape(D2.x[0:12], [3, 4])

    F_new = lab3.fmatrix_from_cameras(C1_refined, C2)

    return F_new