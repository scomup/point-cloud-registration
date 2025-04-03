import numpy as np
import q3dviewer as q3d
from point_cloud_registration import voxel_filter
points, _ = q3d.load_las("/home/liu/Downloads/kyobashi/B-01.las")
mean = np.mean(points['xyz'].astype(np.float64), axis=0)
points['xyz'] = points['xyz'] - mean[np.newaxis,:]

q3d.save_pcd(points, "data/B-01.pcd")
