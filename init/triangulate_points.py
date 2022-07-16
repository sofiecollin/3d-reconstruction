from ext import lab3
import numpy as np

def triangulate_points(view1, view2, K, corr1, corr2):
    C1 = np.array(K @ view1.get_external_mtx())
    C2 = np.array(K @ view2.get_external_mtx())

    tri_points = []

    for i in range(corr1.shape[1]):
        p = lab3.triangulate_optimal(C1, C2, corr1[:, i], corr2[:, i])
        tri_points.append(p)

    tri_points = np.asarray(tri_points)

    return tri_points