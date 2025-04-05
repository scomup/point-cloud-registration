from point_cloud_registration import ICP, PlaneICP, NDT, VPlaneICP

from point_cloud_registration import makeRt, expSO3, transform_points, makeT, color_by_voxel
import numpy as np
from benchmark.test_data import generate_test_data


try:
    import q3dviewer as q3d
except ImportError:
    print("Please install q3dviewer using 'pip install q3dviewer==1.1.4'")
    exit(0)


class CMMViewer(q3d.Viewer):
    """
    This class is a subclass of Viewer, which is used to create a cloud movie maker.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_items({
            'grid': q3d.GridItem(size=10, spacing=1),
            'map': q3d.CloudItem(size=2, alpha=0.5, point_type='PIXEL', \
                                  color_mode='FLAT', color='lime', depth_test=True),
            'scan': q3d.CloudItem(size=0.05, alpha=1, point_type='SPHERE', \
                                  color_mode='FLAT', color='r', depth_test=True),
            'norm': q3d.LineItem(width=2, color='lime', line_type='LINES')})

        self.map, self.scan_org = generate_test_data(t=np.array([0, 0, 0]))
        self.scan = self.scan_org.copy()
        self['map'].set_data(self.map)
        self['scan'].set_data(self.scan)

    def add_control_panel(self, main_layout):
        """
        Add a control panel to the viewer.
        """
        # Create a vertical layout for the settings
        setting_layout = q3d.QVBoxLayout()

        # Add a label for the settings
        setting_layout.addWidget(q3d.QLabel("Select Matching method:"))
        self.combo_items = q3d.QComboBox()
        self.combo_items.addItems(['ICP', 'PlaneICP', 'NDT', 'VPlaneICP'])
        self.combo_items.setCurrentIndex(3)
        self.combo_items.setToolTip("Select the matching method")
        self.combo_items.setStyleSheet("QComboBox { background-color: lightgray; }")
        self.combo_items.currentIndexChanged.connect(self.update_method)
        setting_layout.addWidget(self.combo_items)
        self.box_k = q3d.QSpinBox()
        self.box_k.setRange(5, 30)
        self.box_k.setValue(5)
        self.box_k.setPrefix("kdtree max neighbour: ")
        setting_layout.addWidget(self.box_k)
        self.box_v = q3d.QDoubleSpinBox()
        self.box_v.setRange(0.5, 2)
        self.box_v.setValue(1)
        self.box_v.setSingleStep(0.1)
        self.box_v.setDecimals(1)
        self.box_v.setPrefix("Voxel size: ")
        self.box_v.valueChanged.connect(self.update_voxel_size)
        setting_layout.addWidget(self.box_v)


        # Add XYZ spin boxes
        setting_layout.addWidget(q3d.QLabel("Set inital XYZ:"))
        self.box_x = q3d.QDoubleSpinBox()
        self.box_x.setSingleStep(0.01)
        setting_layout.addWidget(self.box_x)
        self.box_y = q3d.QDoubleSpinBox()
        self.box_y.setSingleStep(0.01)
        setting_layout.addWidget(self.box_y)
        self.box_z = q3d.QDoubleSpinBox()
        self.box_z.setSingleStep(0.01)
        setting_layout.addWidget(self.box_z)
        max_trans = 0.5
        self.box_x.setRange(-max_trans, max_trans)
        self.box_y.setRange(-max_trans, max_trans)
        self.box_z.setRange(-max_trans, max_trans)

        # Add RPY spin boxes
        setting_layout.addWidget(q3d.QLabel("Set inital Roll-Pitch-Yaw:"))
        self.box_roll = q3d.QDoubleSpinBox()
        self.box_roll.setSingleStep(0.01)
        setting_layout.addWidget(self.box_roll)
        self.box_pitch = q3d.QDoubleSpinBox()
        self.box_pitch.setSingleStep(0.01)
        setting_layout.addWidget(self.box_pitch)
        self.box_yaw = q3d.QDoubleSpinBox()
        self.box_yaw.setSingleStep(0.01)
        setting_layout.addWidget(self.box_yaw)
        max_range = np.pi / 180. * 10
        self.box_roll.setRange(-max_range, max_range)
        self.box_pitch.setRange(-max_range, max_range)
        self.box_yaw.setRange(-max_range, max_range)

        self.box_x.valueChanged.connect(self.update_transform)
        self.box_y.valueChanged.connect(self.update_transform)
        self.box_z.valueChanged.connect(self.update_transform)
        self.box_roll.valueChanged.connect(self.update_transform)
        self.box_pitch.valueChanged.connect(self.update_transform)
        self.box_yaw.valueChanged.connect(self.update_transform)

        setting_layout.addStretch()
        # Add the settings layout to the main layout
        main_layout.addLayout(setting_layout)

    def update_voxel_size(self):
        """
        Update the voxel size based on the spin box value.
        """
        voxel_size = self.box_v.value()
        map_color = color_by_voxel(self.map, voxel_size)
        self['map'].set_color_mode('RGB')
        self['map'].set_data(map_color)
        
    def update_method(self):
        """
        Update the matching method based on the selected index.
        """
        index = self.combo_items.currentIndex()
        if index == 0:
            self.box_v.setHidden(True)
            self.box_k.setHidden(True)
            self.method = ICP()
        elif index == 1:
            self.box_v.setHidden(True)
            self.box_k.setHidden(False)
            # k = self.box_k.value()
            self.method = PlaneICP()
        elif index == 2:
            self.box_k.setHidden(True)
            self.box_v.setHidden(False)
            voxel_size = self.box_v.value()
            self.method = NDT(voxel_size=voxel_size)
        elif index == 3:
            self.box_k.setHidden(True)
            self.box_v.setHidden(False)
            voxel_size = self.box_v.value()
            # k = self.box_k.value()
            self.method = VPlaneICP(voxel_size=voxel_size)
        else:
            raise ValueError("Invalid method selected.")
        self.method.set_target(self.map, k=self.box_k.value(), voxel_size=self.box_v.value())

    def update_transform(self):
        """
        Update the transformation matrix based on the spin box values.
        """
        x = self.box_x.value()
        y = self.box_y.value()
        z = self.box_z.value()
        roll = self.box_roll.value()
        pitch = self.box_pitch.value()
        yaw = self.box_yaw.value()

        # Create the transformation matrix
        R = q3d.euler_to_matrix(np.array([roll, pitch, yaw]))
        t = np.array([x, y, z])
        T = makeT(R, t)
        self.scan = transform_points(T, self.scan_org)
        # Apply the transformation to the cloud item
        self['scan'].set_data(self.scan)

if __name__ == '__main__':

    # Generate N x 3 points
    # icp = PlaneICP(voxel_size=1.0, max_iter=30, max_dist=100, tol=1e-5)
    # icp.set_target(map)
    # T_new = icp.align(scan, init_T=T, verbose=True)
    # # icp.max_dist = 0.1
    # # T_new = icp.align(scan, init_T=T_new, verbose=True)
    # R_new, t_new = makeRt(T_new)
# 
    # scan_new = transform_points(T_new, scan)
# 
    # print(T_new)
    # scan_new = (R_new @ scan.T).T + t_new

    app = q3d.QApplication([])

    # create viewer
    viewer = CMMViewer(name='example')
    viewer.show()
    app.exec()

