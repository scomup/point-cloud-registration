from point_cloud_registration.voxel import VoxelGrid, color_by_voxel
import numpy as np
from benchmark.test_data import generate_test_data
try:
    import q3dviewer as q3d
except ImportError:
    print("To visualize the results, please install q3dviewer first by using 'pip install q3dviewer==1.1.4'")
    exit(1)


if __name__ == '__main__':

    points, _ = generate_test_data()
    voxel_size = 2
    voxels = VoxelGrid(voxel_size)
    import time
    t1 = time.time()
    voxels.set_points(points)
    t2 = time.time()
    voxels.query(points, ['mean', 'norm'])
    t3 = time.time()
    print(f"add points time: {(t2 - t1) * 1000:.2f} ms")
    print(f"find points time: {(t3 - t2) * 1000:.2f} ms")
    app = q3d.QApplication([])

    # create viewer
    viewer = q3d.Viewer(name='example')
    # add all items to viewer
    viewer.add_items({
        'grid': q3d.GridItem(size=10, 
                             spacing=1),
        'scan': q3d.CloudItem(size=2, 
                              alpha=0.5, 
                              point_type='PIXEL', 
                              depth_test=True),
        'norm': q3d.LineItem(width=2, 
                             color='lime', 
                             line_type='LINES')})
    
    color_points = color_by_voxel(points, voxel_size)
    viewer['scan'].set_data(color_points)

    normal_length = 0.5
    lines = []
    for mean, norm in zip(voxels.mean, voxels.norm):
            lines.append(mean)
            lines.append(mean + norm * normal_length)
    lines = np.array(lines)
    viewer['norm'].set_data(lines)

    viewer.show()
    app.exec()
