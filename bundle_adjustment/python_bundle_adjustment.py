
from ext import lab3
from classes.P import P
from classes.Q import Q
from classes.PQ_Table import PQ_Table
import numpy as np
from scipy.optimize import least_squares

def cost_func(x0, pq_table, nr_views, nr_points_3D, K, x0_shape):

    x0 = np.reshape(x0,x0_shape)
    if type(x0) is not np.matrix:
        x0 = x0.view(dtype = np.matrix)

    R_arr, t_arr, p_arr = extract_data(x0, nr_views, nr_points_3D)
    e_squared = 0

    diff_arr = np.array([])
    for i in range(nr_points_3D):
        for j in range(nr_views):
            C_mtx = shape_C_mtx(K,R_arr[j],t_arr[j])
            p_proj = np.dot(C_mtx, lab3.homog(p_arr[i]))
            p_proj_norm = p_proj/p_proj[2]
            diff = pq_table[j,i,:] - p_proj_norm[0:2].T
            diff_arr = np.append(diff_arr,diff)

    return diff_arr

def cost_func2(x0, Keeper, nr_views, nr_points_3D, K, x0_shape):

    x0 = np.reshape(x0,x0_shape)
    if type(x0) is not np.matrix:
        x0 = x0.view(dtype = np.matrix)

    R_arr, t_arr, p_arr = extract_data(x0, nr_views, nr_points_3D)
    e_squared = 0

    diff_arr = np.array([])
    q = 0
    for i in range(len(Keeper.views)):
        for j in range(len(Keeper.points3d)):
            
            for k in range(len(Keeper.points3d[j].observations)):
                if Keeper.views[i] is Keeper.points3d[j].observations[k].view:
                    q = q + 1
                    print('view, 3dpoint, obs: ',i, j, k)
                    C_mtx = shape_C_mtx(K,R_arr[i],t_arr[i])
                    p_proj = np.dot(C_mtx, lab3.homog(p_arr[j]))
                    p_proj_norm = p_proj/p_proj[2]
                    diff = Keeper.points3d[j].observations[k].point_2d - p_proj_norm[0:2].T
                    diff_arr = np.append(diff_arr,diff)

    return diff_arr, q

def shape_C_mtx(K, R, t):
    return np.dot(K, np.concatenate((R, t), axis = 1))

def extract_data(x0, nr_views, nr_points_3D):
    x = x0.reshape(4*nr_views+nr_points_3D, 3)
    R_arr = []
    for i in range(nr_views):
        R = x[i*3:i*3+3]
        R_arr.append(R)
    t_arr = []
    for i in range(nr_views*3,nr_views*4):
        t = x[i].T
        t_arr.append(t)
    p_arr = []
    for i in range(nr_views*4,nr_points_3D+nr_views*4):
        p = x[i].T
        p_arr.append(p)

    return R_arr, t_arr, p_arr

# OK, i hope
def assemble_data(Book):
    x0 = None
    for i in range (len(Book.views)):
        if x0 is None:
            x0 = Book.views[i].R
        else:
            x0 = np.concatenate((x0, Book.views[i].R), axis = 0)
    
    for i in range (len(Book.views)):
        x0 = np.concatenate((x0, Book.views[i].t.T), axis = 0)
    
    for i in range (len(Book.points3d)):
        x0 = np.concatenate((x0, np.array([Book.points3d[i].coordinates])), axis = 0)

    return x0, np.shape(x0)

def update_data(x, P, Q, pq_table, x0_shape):

    x = np.reshape(x, x0_shape)
    x = x.view(dtype = np.matrix)
    R_arr, t_arr, p_arr = extract_data(x, len(Q.views), len(P.points_3d))
    R_arr = np.asarray(R_arr)
    t_arr = np.asarray(t_arr)
    p_arr = np.asarray(p_arr)

    for i in range (len(Q.views)):
        Q.views[i].R = R_arr[i]
        Q.views[i].t = t_arr[i]
    
    for i in range (len(P.points_3d)):
        P.points_3d[i] = p_arr[i].T

    return

def python_bundle_adjustment(P,Q,pq_table, nr_views, nr_points_3D, K):
    x0, x0_shape = assemble_data(P,Q)
    x0_flat = np.asarray(x0).ravel()
    e = 0.0000001
    bounds1 = x0_flat[0:12]-e
    bounds2 = x0_flat[0:12]+e
    bounds1 = np.append(bounds1, [-np.inf]*(len(x0_flat)-12))
    bounds2 = np.append(bounds2, [np.inf]*(len(x0_flat)-12))
    x0_bounds = (bounds1, bounds2)
    res = least_squares(cost_func, x0_flat, bounds=x0_bounds, args=(pq_table.get_mtx(), 2, 37, K, x0_shape))
    # update_data(res.x, P, Q, pq_table, x0_shape)
    return res

# Example for calling. OBS, nr_views and nr_points are here hard coded. 
# These arguments might later be removed. 
# python_bundle_adjustment(P,Q,pq_table,2,37,K)