import numpy as np
import time
import open3d as o3d
from point_cloud_registration import estimate_normals, get_norm_lines
from point_cloud_registration.kdtree import KDTree
import q3dviewer as q3d

from benchmark.test_data import generate_test_data



def test_our_estimate_normals(points, k):
    start_time = time.time()
    normals = estimate_normals(points, k)
    elapsed_time = time.time() - start_time
    return elapsed_time, normals


def test_open3d_estimate_normals(points, k):
    start_time = time.time()
    # Convert points to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Estimate normals using Open3D
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    normals = np.asarray(pcd.normals)
    elapsed_time = time.time() - start_time
    return elapsed_time, normals


def get_norm_lines(points, normals, length=0.1):
    """
    Generate lines for visualization of normals.
    """
    offset_points = points + normals * length
    lines = np.empty((2 * points.shape[0], points.shape[1]), dtype=points.dtype)
    lines[::2] = points
    lines[1::2] = offset_points
    return lines


if __name__ == '__main__':
    points, _ = generate_test_data()
    k = 5

    # Test our implementation
    t1, our_normals = test_our_estimate_normals(points, k)

    # Test Open3D implementation
    t2, open3d_normals = test_open3d_estimate_normals(points, k)

    # from point_cloud_registration.estimate_normals import estimate_normals_with_tree, estimate_normals_with_tree2
    # tree = KDTree(points)
    # t1 = time.time()
    # estimate_normals_with_tree(points, tree, k=k)
    # t2 = time.time()
    # estimate_normals_with_tree2(points, tree, k=k)
    # t3 = time.time()
    # print("KDTree time:", t2 - t1)
    # print("KDTree2 time:", t3 - t2)

    # Print results
    print("\nComparison:")
    print(f"Our estimate_normals time: {t1:.6f} seconds")
    print(f"Open3D estimate_normals time: {t2:.6f} seconds")

    # Optional: Visualize the results using q3dviewer
    app = q3d.QApplication([])
    viewer = q3d.Viewer(name='Normals Comparison')
    viewer.add_items({
        'points': q3d.CloudItem(size=0.1, alpha=1, point_type='SPHERE', color_mode='FLAT', color='#ff0000', depth_test=True),
        'normals': q3d.LineItem(width=2, color='#00ff00', line_type='LINES'),
    })
    viewer['points'].set_data(points)

    norm_line = get_norm_lines(points, our_normals, length=0.05)
    viewer['normals'].set_data(norm_line)
    viewer.show()
    app.exec()
