import numpy as np
import open3d as o3d

# 读取txt，xyz格式
# pcd = o3d.io.read_point_cloud('123.txt', format='xyz')#pcd_data.path
# o3d.visualization.draw_geometries([pcd])

# 读取pcd格式
pcd = o3d.io.read_point_cloud('small.pcd', format='pcd')
o3d.visualization.draw_geometries([pcd])


