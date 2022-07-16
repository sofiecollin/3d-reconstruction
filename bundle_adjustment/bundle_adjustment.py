import numpy as np
from build.ceres_python import ceres_BA
from classes.BookKeeping import *
from scipy.spatial.transform import Rotation as Rot

def update_bookkeeping(Keeper, args, n_views, n_points, len_camera):
    # Update the bookkeeping by first converting back to rotation matrix
    if len_camera == 6:
        for i in range(n_views):
            r, _ = cv2.Rodrigues(args[i*len_camera:i*len_camera+3])
            t = args[i*len_camera+3:i*len_camera+len_camera].reshape(3,1)
            Keeper.views[i].set_pose(r,t)
            start_points = n_views*len_camera
            for j in range(n_points):
                coord = args[start_points+j*3:start_points+j*3+3]
                Keeper.points3d[j].coordinates = coord
    else:
        raise ValueError("Camera size not acceptable")
        
def bundle_adjustment(Keeper, K):
    # Initiate parameters
    observations = Keeper.observations
    views = Keeper.views
    points = Keeper.points3d
    n_observations = len(observations)
    n_views = len(views)
    n_points = len(points)
    len_camera = 6

    args = np.zeros(n_views*len_camera+n_points*3) # Flattened array of all cameras and 3d points
    observations_2d = np.zeros(n_observations*2)
    camera_index = np.zeros(n_observations)
    point_index = np.zeros(n_observations)

    # Fill observations for ceres
    for i, observation in enumerate(observations):
        
        observations_2d[i * 2 + 0] = observation.point_2d[0]
        observations_2d[i * 2 + 1] = observation.point_2d[1]

        # For ceres to know which camera/point in the array it should pick for the observation
        camera_index[i] = views.index(observation.view)
        point_index[i] = points.index(observation.point_3d)

     # Fill args(all cameras and 3d points in array form) for ceres
    # Cameras
    for i, view in enumerate(views):
        # Quaternions
        if len_camera == 7:
            args[i*len_camera:i*len_camera+4] = np.ravel(view.get_rot_as_quat().flatten('F')[:, None])
            args[i*len_camera+4:i*len_camera+len_camera] = np.ravel(view.t)
        # Angle axis
        elif len_camera == 6:
            args[i*len_camera:i*len_camera+3] = np.ravel(view.get_rot_as_rodrigues().flatten('F')[:, None])
            args[i*len_camera+3:i*len_camera+len_camera] = np.ravel(view.t)
        else:
            raise ValueError("Camera size not acceptable")
        
    # 3d points
    point_counter = 0
    for i in range(n_views*len_camera, n_views*len_camera+n_points):
        coords = points[point_counter].coordinates
        
        args[n_views*len_camera+3*point_counter] = coords[0]
        args[n_views*len_camera+1+3*point_counter] = coords[1]
        args[n_views*len_camera+2+3*point_counter] = coords[2]
        point_counter += 1
    
    args_old = args.copy()

    K[0,1] = 0
    ceres_BA(n_views, n_points, n_observations, args,  observations_2d,  camera_index,  point_index, K)

    print("Difference from before BA: " + str(np.sum(args_old-args)))
    
    update_bookkeeping(Keeper, args, n_views, n_points, len_camera)

    return None 