from ext import lab3
from init.calc_fmatrix_ransac import calc_fmatrix_ransac
from init.calc_fmatrix_gs import calc_fmatrix_gs
from init.calc_ematrix import calc_ematrix
from init.extract_pose import extract_pose
from init.setup_views import setup_views
#from _tests_.test_ceres import test_ceres
from init.get_correspondences import get_correspondences
from init.triangulate_points import triangulate_points
from _tests_.test_init import test_init
from _tests_.test_pose_extraction import test_pose_extraction
from bundle_adjustment.bundle_adjustment import bundle_adjustment
from classes.P import P
from classes.Q import Q
from classes.PQ_Table import PQ_Table
import numpy as np
import matplotlib.pyplot as plt
from classes.BookKeeping import *
from add_views.add_views import add_view
from visualization import visualization
from eval.eval import eval_poses
import cv2

# There are 37 images and 36 ground truths?

Keeper = BookKeeper()

(views, K) = setup_views() # Use general K since it is same for all views.
# K have some sign errors. Hardcode until fixed.
#K = np.matrix([[-3.2173*1000, 0.0786*1000, 0.2899*1000], [0, -2.2924*1000, -1.0705*1000], [0, 0, 0.0010*1000]])
(corr1, corr2) = get_correspondences(views[0], views[1])

#lab3.show_corresp(views[0].image, views[1].image, corr1, corr2)
#plt.show()

(F_ransac, inliers1, inliers2) = calc_fmatrix_ransac(corr1, corr2, n_iterations=100, threshold=0.1)
# Note! Inliers should be all inliers, not just the 8 best. Should be fixed when manual correpondences are found.
F = calc_fmatrix_gs(F_ransac, inliers1, inliers2)

#plt.figure(1)
#plt.imshow(views[0].image)
#lab3.plot_eplines(F, inliers2, (views[0].image.shape[1], views[0].image.shape[0]))

#plt.figure(2)
#plt.imshow(views[1].image)
#lab3.plot_eplines(F.T, inliers1, (views[1].image.shape[1], views[1].image.shape[0]))

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

Keeper.add_view(views[0])
Keeper.add_view(views[1])

C1 = np.array(K @ views[0].get_external_mtx())
C2 = np.array(K @ views[1].get_external_mtx())
for i in range(corr1.shape[1]):
    point3d_coords = lab3.triangulate_optimal(C1, C2, corr1[:, i], corr2[:, i])
    point3d = Point3D(point3d_coords)
    Keeper.add_point3d(point3d)
    Keeper.bookkeep_observation(views[0], corr1[:, i], point3d)
    Keeper.bookkeep_observation(views[1], corr2[:, i], point3d)

### TEST CASE 2 ###
test_pose_extraction(views[0], views[1], K, corr1, corr2, Keeper.points3d) # Add more arguments if needed
####################

bundle_adjustment(Keeper, K)

# Loop and add up to view number view_nr
view_nr = 35

for i in range(1,view_nr):
    prev_view = views[i]
    new_view = views[i+1]
    add_view(prev_view, new_view, K, Keeper)
    bundle_adjustment(Keeper, K)

# project and plot by open cv
'''
points_3d_temp = []
for point3d in Keeper.points3d:
    points_3d_temp.append(point3d.coordinates)

points_2d_temp,_ = cv2.projectPoints(np.matrix(points_3d_temp), views[view_nr].R,views[view_nr].t, K, np.zeros((1,4)))
plt.imshow(Keeper.views[view_nr].image)
plt.scatter(points_2d_temp[:,0][:,0],points_2d_temp[:,0][:,1], c='r', s=40)
plt.show()
'''
# Plot projected 2d points

plt.imshow(Keeper.views[view_nr].image)
for point3d in Keeper.points3d:
    P1 = K @ Keeper.views[view_nr].get_external_mtx()
    point3dCoord = np.append(point3d.coordinates, 1)
    proj1 = P1 @ point3dCoord # 2D Projection using K(R|T)X
    proj1 = (proj1 / proj1[0, 2])[:, 0:2] # Convert from homogeneous

    plt.scatter(proj1[0,0], proj1[0,1], c='r', s=40)
plt.show()

all_poses = []
for view in views:
    all_poses.append(view.get_external_mtx())

e_pos_norm, e_ang_norm = eval_poses(np.array(all_poses))
print(e_pos_norm, e_ang_norm)

#Visualize 3d points through first view (you can move around)
#visualization.vis_3d_points(Keeper.points3d, Keeper.views[0].image)
visualization.vis_3d_camera_and_points(Keeper.views, Keeper.points3d)

