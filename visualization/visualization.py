import copy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib._png import read_png
from matplotlib.cbook import get_sample_data

def vis_2d_points(image, points):
  fig, ax = plt.subplots()
  imshow_args = {'interpolation': 'nearest'}
  ax.imshow(image, **imshow_args)

  ax.set_xlim(0, image.shape[1]-1)
  ax.set_ylim(image.shape[0]-1, 0)

  ax.plot(points[0], points[1], 'o', color='firebrick')
  plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,
                          wspace=0.05, hspace=0.05)
  return fig

#TODO This one uses matplotlib but it's very slow
def vis_3d_scatter(image, points): 

  coordinates = np.zeros((len(points), 3))

  for point in range(len(points)):
    coordinates[point, :] = points[point].coordinates

  fig = plt.figure()
  ax = fig.gca(projection='3d')

  coordinates = coordinates.T

  surf = ax.scatter(coordinates[0], coordinates[1], coordinates[2])

  ax.set_xlim(0, image.shape[1]-1)
  ax.set_ylim(image.shape[0]-1, 0)

  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  # 10 is equal length of x and y axises of your surface
  stepX, stepY = 10. / image.shape[0], 10. / image.shape[1]

  X1 = np.arange(-5, 5, stepX)
  Y1 = np.arange(-5, 5, stepY)
  X1, Y1= np.meshgrid(X1, Y1)
  Z1 = X1*0

  # stride args allows to determine image quality 
  # stride = 1 work slow
  ax.plot_surface(X1, Y1, Z1, rstride=1, cstride=1)

  plt.show()

# Open 3D 
def vis_3d_points(points, image=[]):
    
    coordinates = np.zeros((len(points), 3))
    print(coordinates.shape)

    for point in range(len(points)):
      coordinates[point, :] = points[point].coordinates

    #print('xyz')
    #print(coordinates)

    #TODO Add image overlay
    if image.shape[0] > 0:
      image2d = o3d.geometry.Image(image)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinates)
    o3d.io.write_point_cloud("visualization/sync.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("visualization/sync.ply")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()

    pcd_load.estimate_normals()
    pcd_load.orient_normals_towards_camera_location()

    vis.add_geometry(image2d)
    vis.add_geometry(pcd_load)

    vis.run()
    vis.destroy_window()

def vis_3d_points_anim(points): 
    print('xyz')
    print(points)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("visualization/sync.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("visualization/sync.ply")
    pcd_load.normalize_normals()
    o3d.visualization.draw_geometries([pcd_load])

def vis_3d_camera_and_points(views, points):
    # Visualize Camera and Points
    # views: send in views from BK
    # points: send in points from BK

    # Producing coordinate system axis in increasing order
    # X has smallest gap between points, Y has next smallest and Z has largest gap
    #points_axis = [[0,0,0],[0.4,0,0],[0.9,0,0], [1.3,0,0] ,[1.7,0,0], [2.1,0,0], [2.4,0,0], [0,1,0], [0,2,0], [0,3,0], [0,4,0], [0,5,0], [0,0,3], [0,0,7], [0,0,10], [0,0,13]]

    cam = np.zeros((len(views), 3))

    points_3d = np.zeros((len(points), 3))
    points_color = np.zeros((len(points), 3))

    for view in range(len(views)):
      cam[view, :] = (-views[view].R.T @ views[view].t).T
    
  
    # Plot axis
    '''
    counter = 0
    for point in range(len(points)+len(points_axis)):
      if point < len(points):
        points_3d[point, :] = points[point].coordinates
      else:
        points_3d[point, :] = points_axis[counter]
        counter += 1
    '''

    for point in range(len(points)):
      point_color = np.zeros((len(points[point].observations), 3))

      # If their corresponding color would be somewhat incorrect
      for obs in range(len(points[point].observations)):
        color_ind = np.round(points[point].observations[obs].point_2d).astype(int)
        point_color[obs, :] = points[point].observations[obs].view.image[color_ind[1], color_ind[0]]

      points_3d[point, :] = points[point].coordinates
      # Take the average of all the observations
      points_color[point, :] = np.sum(point_color, axis=0) / len(points[point].observations)

    print(len(points[3].observations))

    #TODO Add image overlay
    image = views[0].image
    if image.shape[0] > 0:
      image2d = o3d.geometry.Image(image)
  

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd_points = o3d.geometry.PointCloud()
    pcd_cam = o3d.geometry.PointCloud()

    pcd_points.points = o3d.utility.Vector3dVector(points_3d)
    points_color = (points_color - 128) / 128
    pcd_points.colors = o3d.utility.Vector3dVector(points_color)
    #pcd_load_points.voxel_down_sample(voxel_size=0.5)
    pcd_points.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=3, max_nn=100))
    pcd_points.orient_normals_towards_camera_location(cam[0, :])


    pcd_cam.points = o3d.utility.Vector3dVector(cam)
    
    # Use sync_points.ply to import to MeshLab
    o3d.io.write_point_cloud("visualization/sync_points.ply", pcd_points)
    
    o3d.io.write_point_cloud("visualization/sync_cam.ply", pcd_cam)

    # Load saved point cloud and visualize it
    pcd_load_points = o3d.io.read_point_cloud("visualization/sync_points.ply")

    pcd_load_cam = o3d.io.read_point_cloud("visualization/sync_cam.ply")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
  

  
    pcd_load_cam.estimate_normals()
    pcd_load_cam.orient_normals_towards_camera_location()

    npts = np.concatenate((pcd_load_points.points, pcd_load_points.normals), axis=1).astype('float32')

    np.savetxt('test.npts', npts, delimiter=' ')

    #vis.add_geometry(image2d)
    vis.add_geometry(pcd_load_points)
    vis.add_geometry(pcd_load_cam)

    vis.run()
    vis.destroy_window()