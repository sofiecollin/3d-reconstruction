import numpy as np
from scipy import io
import cv2
#from estRT import estRT

# Get's the dinos ground truth data.
def get_gt():
    dino_gt = io.loadmat('dataset/BADino2.mat')
    cams_gt = dino_gt['newPs']

    for i in range(np.shape(cams_gt)[1]):
        cam = cams_gt[0,i]
        if i is 0:
            K_gt, R_gt, t_gt = cv2.decomposeProjectionMatrix(cam)[0:3]
            t_gt = t_gt/t_gt[3]
        else:
            K, R, t = cv2.decomposeProjectionMatrix(cam)[0:3]
            t = t/t[3]
            K_gt = np.append(K_gt, K)
            R_gt = np.append(R_gt, R)
            t_gt = np.append(t_gt, t)

    K_gt = np.reshape(K_gt,(np.shape(cams_gt)[1],3,3))
    R_gt = np.reshape(R_gt,(np.shape(cams_gt)[1],3,3))
    t_gt = np.reshape(t_gt,(np.shape(cams_gt)[1],4,1))
    
    return K_gt, R_gt, t_gt


# Takes an estimate of RT in np.array, 3x4 format. 
# Returns gt and est of R and t in correct format for eval calculations
def get_gt_and_est(Rt_est):
    K_gt, R_gt, t_gt = get_gt()
    R_gt = R_gt[0:36,:,:]
    t_gt = t_gt[0:36,0:3,:]

    est = Rt_est
    a, b, t_est = np.array_split(est,3,axis=2)
    R_est = np.concatenate((a,b),axis = 2)

    return R_gt, t_gt, R_est, t_est

# Calculates centroid of a vector of vectors
def calc_centroids(t_vec):
    nr_points = len(t_vec)
    coord_sum = np.array([[0],[0],[0]])
    for i, coords in enumerate(t_vec, 2):
        coord_sum = coord_sum + coords
    centroid = coord_sum/nr_points

    return centroid

# Estimates the rigid transformation between two sets of vectors
def est_rig_trans(R1, t1, R2, t2):
    a0 = calc_centroids(t2)
    b0 = calc_centroids(t1)
    a_c = t2-a0
    b_c = t1-b0

    a_scale = est_scale(a_c)
    b_scale = est_scale(b_c)
    scale_factor = b_scale / a_scale
    a_cs = a_c * scale_factor
    
    a_shape = np.shape(a_c)
    A = np.reshape(a_c, (3,len(a_c)))
    B = np.reshape(b_c, (3,len(b_c)))

    R_global = OPP(A,B)
    t_map = b_c - R_global @ a_cs

    t_mapped = scale_factor*R_global @ t2 + t_map
    R_mapped = R2 @ R_global.T
    
    return t_mapped, R_mapped

# Estimates the scaling between two sets of centered vectors
def est_scale(t_vec_centered):
    nr_points = len(t_vec_centered)
    summed_distance = 0
    for i, coords in enumerate(t_vec_centered,2):
        summed_distance = summed_distance + np.linalg.norm(coords)
    scale = np.sqrt(summed_distance/nr_points)
    
    return scale

# Finds a rotation matrix R between to centered and scaled sets A and B
def OPP(A, B):
    U, S, V = np.linalg.svd(A @ B.T)
    R = V @ U       # note: numpy's V is already transposed
    if abs(abs(np.linalg.det(R))-1) < 0.001:
        return R
    else:
        raise ValueError("determinant of rotation matrix R is not close to 1. Cannot perform OPP")

# Calculates the error vectors of two camera sets positions and rotations
def calc_camera_errors(t_gt, t_mapped, R_gt, R_mapped):
    e_pos = np.abs(t_gt-t_mapped)
    e_angle = 2*np.arcsin(np.abs(R_mapped-R_gt)/np.sqrt(8))

    return e_pos, e_angle

# Calculates the normed errors of positional and angle errors
def calc_normed_errors(e_pos, e_angle):
    e_norm_pos = np.linalg.norm(e_pos)/len(e_pos)
    e_norm_angle = np.linalg.norm(e_angle)/len(e_pos)
    return e_norm_pos, e_norm_angle

# Evaluates the hard coded test set with ground truth
def eval_test():
    est = estRT
    R_gt, t_gt, R_est, t_est = get_gt_and_est(est)

    t_mapped, R_mapped = est_rig_trans(R_gt, t_gt, R_est, t_est)
    e_pos, e_angle = calc_camera_errors(t_gt, t_mapped, R_gt, R_mapped)
    e_pos_norm, e_angle_norm = calc_normed_errors(e_pos, e_angle)

    return e_pos_norm, e_angle_norm

# Evaluates estimated camera positions and rotations and returns the normed errors
# between the sets positions and rotations
def eval_poses(est):
    R_gt, t_gt, R_est, t_est = get_gt_and_est(est)
    t_mapped, R_mapped = est_rig_trans(R_gt, t_gt, R_est, t_est)
    e_pos, e_angle = calc_camera_errors(t_gt, t_mapped, R_gt, R_mapped)
    e_pos_norm, e_angle_norm = calc_normed_errors(e_pos, e_angle)

    return e_pos_norm, e_angle_norm