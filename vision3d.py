import numpy as np
import open3d as o3d

#pcd_data = o3d.data.PLYPointCloud()
#pcd = o3d.io.read_point_cloud("../../test_data/my_points.txt", format='xyz')

pcd = o3d.io.read_point_cloud('C:\\Users\\zyh\\Desktop\\3\\hm3.txt', format='xyz')#pcd_data.path
o3d.visualization.draw_geometries([pcd])