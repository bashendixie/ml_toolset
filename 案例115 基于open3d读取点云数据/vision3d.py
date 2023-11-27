import numpy as np
import open3d as o3d

#pcd_data = o3d.data.PLYPointCloud()
#pcd = o3d.io.read_point_cloud("../../test_data/my_points.txt", format='xyz')

# pcd = o3d.io.read_point_cloud('123.txt', format='xyz')#pcd_data.path
# o3d.visualization.draw_geometries([pcd])

# pcd=o3d.io.read_point_cloud("input.pcd")


pcd = o3d.io.read_point_cloud('C:\\Users\\zyh\\Desktop\\100hz.pcd', format='pcd')#pcd_data.path
o3d.visualization.draw_geometries([pcd])


# pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
#                                          ransac_n=3,
#                                          num_iterations=10000)
# [a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
# print("Displaying pointcloud with planar points in red ...")
# inlier_cloud = pcd.select_by_index(inliers)
# inlier_cloud.paint_uniform_color([1.0, 0, 0])
# outlier_cloud = pcd.select_by_index(inliers, invert=True)
# o3d.visualization.draw([inlier_cloud, outlier_cloud])