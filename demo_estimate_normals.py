import numpy as np
import time
from point_cloud_registration import estimate_normals, get_norm_lines
try:
    import q3dviewer as q3d
except ImportError:
    print("To visualize the results, please install q3dviewer first by using 'pip install q3dviewer'")
    exit(1)

from benchmark.test_data import generate_test_data


def get_norm_lines(points, normals, length=0.1):
    """
    Generate lines for visualization of normals.
    """
    offset_points = points + normals * length
    lines = np.empty(
        (2 * points.shape[0], points.shape[1]), dtype=points.dtype)
    lines[::2] = points
    lines[1::2] = offset_points
    return lines


if __name__ == '__main__':
    points, _ = generate_test_data()
    k = 10

    start_time = time.time()
    normals = estimate_normals(points, k)
    elapsed_time = time.time() - start_time

    print(f"Estimate_normals time: {elapsed_time:.6f} seconds")

    # Optional: Visualize the results using q3dviewer
    app = q3d.QApplication([])
    viewer = q3d.Viewer(name='Normals Comparison')
    viewer.add_items({
        'points': q3d.CloudItem(size=0.1, 
                                alpha=1, 
                                point_type='SPHERE', 
                                color_mode='FLAT', 
                                color='r', 
                                depth_test=True),
        'normals': q3d.LineItem(width=2, 
                                color='lime', 
                                line_type='LINES'),
    })
    viewer['points'].set_data(points)

    norm_line = get_norm_lines(points, normals, length=0.05)
    viewer['normals'].set_data(norm_line)
    viewer.show()
    app.exec()
