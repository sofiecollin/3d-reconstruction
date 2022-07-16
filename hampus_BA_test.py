
from ext import lab3
from init.calc_fmatrix_ransac import calc_fmatrix_ransac
from init.calc_fmatrix_gs import calc_fmatrix_gs
from init.calc_ematrix import calc_ematrix
from init.extract_pose import extract_pose
from init.setup_views import setup_views
# from init.test_ceres import test_ceres
from init.get_correspondences import get_correspondences
from init.triangulate_points import triangulate_points
from _tests_.test_init import test_init
from _tests_.test_pose_extraction import test_pose_extraction
# from bundle_adjustment.bundle_adjustment import bundle_adjustment
from classes.P import P
from classes.Q import Q
from classes.PQ_Table import PQ_Table
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from bundle_adjustment.python_bundle_adjustment import python_bundle_adjustment

# There are 37 images and 36 ground truths?

(views, K) = setup_views() # Use general K since it is same for all views.
# K have some sign errors. Hardcode until fixed.
K = np.matrix([[-3.2173*1000, 0.0786*1000, 0.2899*1000], [0, -2.2924*1000, -1.0705*1000], [0, 0, 0.0010*1000]])

(corr1, corr2) = get_correspondences(views[0], views[1])

lab3.show_corresp(views[0].image, views[1].image, corr1, corr2)
#plt.show()

(F_ransac, inliers1, inliers2) = calc_fmatrix_ransac(corr1, corr2, n_iterations=100, threshold=0.1)
# Note! Inliers should be all inliers, not just the 8 best. Should be fixed when manual correpondences are found.
F = calc_fmatrix_gs(F_ransac, inliers1, inliers2)

plt.figure(1)
plt.imshow(views[0].image)
lab3.plot_eplines(F, inliers2, (views[0].image.shape[1], views[0].image.shape[0]))

plt.figure(2)
plt.imshow(views[1].image)
lab3.plot_eplines(F.T, inliers1, (views[1].image.shape[1], views[1].image.shape[0]))

#plt.show()

E = calc_ematrix(F, K)

### TEST CASE 1 ###
test_init(E, K, corr1, corr2)
####################

R1 = np.eye(3, 3)
t1 = np.zeros((3, 1))
views[0].set_pose(R1, t1)
(R2, t2) = extract_pose(E, K, corr1[:, 0], corr2[:, 0])
views[1].set_pose(R2, t2)

points_3d = triangulate_points(views[0], views[1], K, corr1, corr2)

### TEST CASE 2 ###
test_pose_extraction(views[0], views[1], K, corr1, corr2, points_3d) # Add more arguments if needed
####################

P = P()
P.append_points_3d(points_3d)
Q = Q()
Q.add_view(views[0])
Q.add_view(views[1])

pq_table = PQ_Table()
pq_table.add_row(corr1)
pq_table.add_row(corr2)



# Example for callning the python BA. More details in separate file
python_bundle_adjustment(P,Q,pq_table,2,37,K)