
from ext import lab3
import numpy as np
import random

def calc_fmatrix_ransac(corr1, corr2, n_iterations, threshold):
    best_inlier_count = 0
    best_F = None
    best_std = np.inf

    for i in range(n_iterations):
        random_index = random.sample(range(corr1.shape[1]), 8)

        corr1_rand = corr1[:, random_index]
        corr2_rand = corr2[:, random_index]

        F = lab3.fmatrix_stls(corr1_rand, corr2_rand)

        d = lab3.fmatrix_residuals(F, corr1, corr2)

        inlier_count = 0

        for j in range(d.shape[1]):
            norm = np.linalg.norm(d[:, j])
            if norm < threshold:
                inlier_count = inlier_count + 1
        
        std = np.std(np.linalg.norm(d, axis=0))
        
        if (inlier_count > best_inlier_count) or (inlier_count == best_inlier_count and std < best_std):
            best_std = std
            best_inlier_count = inlier_count
            best_F = F
            best_corr1 = corr1_rand
            best_corr2 = corr2_rand

    #inlier_ratio = best_inlier_count / d.shape[1]
    #print("INLIER RATIO", inlier_ratio)

    # Note! Inliers should be all inliers, not just the 8 best. Should be fixed when manual correpondences are found.
    return (best_F, best_corr1, best_corr2)