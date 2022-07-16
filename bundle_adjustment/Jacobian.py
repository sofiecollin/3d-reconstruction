import numpy as np

def generate_jacobian(args, corr, n_views, n_points):
    # points with pos -1 => not visible

    w = corr > 0
    w = w*1 # convert to int 
    w = w.flatten('F')[:, None]
    
    J = np.ones((n_views*(2*n_points),1))
    J = J*w
    J = J[J != 0][:, None]
    J = np.zeros((J.shape[0], args.shape[0]))
    J_it = np.zeros((J.shape[0], args.shape[0]))

    X = args[(n_views*3*4):None].reshape((3, n_points))

    C = np.ones((1,12)) # A camera
    P = np.ones((1,3)) # 3D point

    visible_points = corr[:, 1::2]
    visible_points = (visible_points > 0) * 1
    print(visible_points)

    for m in range(n_points):

        (i, j) = np.where(visible_points == visible_points[:,m])
        print(i)
    
    # Return visible points in form of a mask, and the jacob pattern.
    return J