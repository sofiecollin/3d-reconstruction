import numpy as np

def test_pose_extraction(view1, view2, K, pts1_2d, pts2_2d, pts_3d):
    for i in range(len(pts_3d)):
        pt_3d = np.append(pts_3d[i].coordinates, 1)

        P1 = K @ view1.get_external_mtx()
        proj1 = P1 @ pt_3d # 2D Projection using K(R|T)X
        proj1 = (proj1 / proj1[0, 2])[:, 0:2] # Convert from homogeneous
        diff1 = proj1 - pts1_2d[:, i]
        assert np.linalg.norm(diff1) < 1e-2

        P2 = K @ view2.get_external_mtx()
        proj2 = P2 @ pt_3d # 2D Projection using K(R|T)X
        proj2 = (proj2 / proj2[0, 2])[:, 0:2] # Convert from homogeneous
        diff2 = proj2 - pts2_2d[:, i]
        assert np.linalg.norm(diff2) < 1e-2