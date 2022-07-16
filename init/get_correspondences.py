from scipy import io
import numpy as np

def get_correspondences(view1, view2):
    valid_indices = (view1.points_2d[0, :] != -1) & (view2.points_2d[0, :] != -1) # Check where x-coordinate isn't -1 in either view

    corr1 = view1.points_2d[:, valid_indices]
    corr2 = view2.points_2d[:, valid_indices]
    return (corr1, corr2)