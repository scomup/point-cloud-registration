import q3dviewer as q3d
from point_cloud_registration.voxel import VoxelGrid, color_by_voxel
import numpy as np

if __name__ == '__main__':

    points, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/1.pcd")
    voxel_size = 0.5
    voxels = VoxelGrid(voxel_size)
    import time
    t1 = time.time()
    voxels.add_points(points['xyz'])
    t2 = time.time()
    voxels.query(points['xyz'], ['mean', 'norm'])
    t3 = time.time()
    print(f"add points time: {(t2 - t1) * 1000:.2f} ms")
    print(f"find points time: {(t3 - t2) * 1000:.2f} ms")
    app = q3d.QApplication([])

    # create viewer
    viewer = q3d.Viewer(name='example')
    # add all items to viewer
    viewer.add_items({
        'grid': q3d.GridItem(size=10, spacing=1),
        'scan': q3d.CloudItem(size=0.01, alpha=0.5, point_type='SPHERE', depth_test=True),
        'norm': q3d.LineItem(width=2, color='#00ff00', line_type='LINES')})
    
    color_points = color_by_voxel(points['xyz'], voxel_size)
    viewer['scan'].set_data(color_points)

    normal_length = 0.1
    lines = []
    for key, cell in voxels.voxels.items():
            lines.append(cell.mean)
            lines.append(cell.mean + cell.norm * normal_length)
    lines = np.array(lines)
    viewer['norm'].set_data(lines)

    viewer.show()
    app.exec()
