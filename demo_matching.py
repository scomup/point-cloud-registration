#!/usr/bin/env python3

from point_cloud_registration import ICP, PlaneICP, NDT, VPlaneICP

from point_cloud_registration import makeRt, expSO3, transform_points, makeT, color_by_voxel
import numpy as np
from benchmark.test_data import generate_test_data
from q3dviewer.Qt.QtWidgets import QDialog, QTextEdit, QSpinBox, QLabel, QVBoxLayout, QGroupBox, QDoubleSpinBox, QPushButton

try:
    import q3dviewer as q3d
except ImportError:
    print("Please install q3dviewer using 'pip install q3dviewer==1.1.6'")
    exit(0)


class DEMOViewer(q3d.Viewer):
    """
    This class is a subclass of Viewer, which is used to create a cloud movie maker.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_items({
            'grid': q3d.GridItem(size=10, spacing=1),
            'map': q3d.CloudItem(size=2,
                                 alpha=0.3,
                                 point_type='PIXEL',
                                 color_mode='RGB',
                                 color='r',
                                 depth_test=False),
            'scan': q3d.CloudItem(size=0.05, alpha=1, point_type='SPHERE',
                                  color_mode='FLAT', color='lime', depth_test=False),
            'norm': q3d.LineItem(width=2, color='lime', line_type='LINES')})

        self.method = None
        self.map, self.scan_org = generate_test_data(t=np.array([0, 0, 0]))
        self.scan = self.scan_org.copy()
        color_points = color_by_voxel(self.map, 2)
        self['map'].set_data(color_points)
        self['scan'].set_data(self.scan)

    def add_control_panel(self, main_layout):
        """
        Add a control panel to the viewer.
        """
        # Create a vertical layout for the settings
        setting_layout = QVBoxLayout()

        # Create a container widget for the layout
        setting_widget = q3d.QWidget()
        setting_widget.setLayout(setting_layout)
        setting_widget.setFixedWidth(250)  # Set the fixed width for the settings panel

        # Add a label for the settings
        group_box = QGroupBox("Matching Settings")
        group_layout = QVBoxLayout()

        group_layout.addWidget(QLabel("Select Matching method:"))
        self.combo_items = q3d.QComboBox()
        self.combo_items.addItems(['ICP', 'PlaneICP', 'NDT', 'VPlaneICP'])
        self.combo_items.setCurrentIndex(3)
        self.combo_items.setToolTip("Select the matching method")
        self.combo_items.setStyleSheet(
            "QComboBox { background-color: lightgray; }")
        self.combo_items.currentIndexChanged.connect(self.update_method)
        group_layout.addWidget(self.combo_items)

        self.box_k = QSpinBox()
        self.box_k.setRange(5, 30)
        self.box_k.setValue(5)
        self.box_k.setPrefix("kdtree max neighbour: ")
        group_layout.addWidget(self.box_k)

        self.box_v = QDoubleSpinBox()
        self.box_v.setRange(0.5, 2)
        self.box_v.setValue(1)
        self.box_v.setSingleStep(0.1)
        self.box_v.setDecimals(1)
        self.box_v.setPrefix("Voxel size: ")
        self.box_v.valueChanged.connect(self.update_voxel_size)
        group_layout.addWidget(self.box_v)

        self.box_max_dist = QDoubleSpinBox()
        self.box_max_dist.setRange(0.1, 3)
        self.box_max_dist.setValue(1)
        self.box_max_dist.setSingleStep(0.1)
        self.box_max_dist.setPrefix("Min distance: ")
        group_layout.addWidget(self.box_max_dist)

        group_box.setLayout(group_layout)
        setting_layout.addWidget(group_box)

        # Add a group for initial pose settings
        group_box_pose = q3d.QGroupBox("Initial Pose Settings")
        group_layout_pose = q3d.QVBoxLayout()
        # Add XYZ spin boxes
        group_layout_pose.addWidget(q3d.QLabel("Set initial XYZ:"))
        self.box_x = QDoubleSpinBox()
        self.box_x.setSingleStep(0.01)
        group_layout_pose.addWidget(self.box_x)
        self.box_y = QDoubleSpinBox()
        self.box_y.setSingleStep(0.01)
        group_layout_pose.addWidget(self.box_y)
        self.box_z = QDoubleSpinBox()
        self.box_z.setSingleStep(0.01)
        group_layout_pose.addWidget(self.box_z)
        max_trans = 0.5
        self.box_x.setRange(-max_trans, max_trans)
        self.box_y.setRange(-max_trans, max_trans)
        self.box_z.setRange(-max_trans, max_trans)
        # Add RPY spin boxes
        group_layout_pose.addWidget(q3d.QLabel("Set initial Roll-Pitch-Yaw:"))
        self.box_roll = QDoubleSpinBox()
        self.box_roll.setSingleStep(0.01)
        group_layout_pose.addWidget(self.box_roll)
        self.box_pitch = QDoubleSpinBox()
        self.box_pitch.setSingleStep(0.01)
        group_layout_pose.addWidget(self.box_pitch)
        self.box_yaw = QDoubleSpinBox()
        self.box_yaw.setSingleStep(0.01)
        group_layout_pose.addWidget(self.box_yaw)
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
        group_box_pose.setLayout(group_layout_pose)
        setting_layout.addWidget(group_box_pose)

        self.button_matching = QPushButton("Matching")
        self.button_matching.setToolTip("Start matching")
        self.button_matching.setStyleSheet(
            "QPushButton { background-color: lightgreen; }")
        self.button_matching.clicked.connect(self.do_matching)
        setting_layout.addWidget(self.button_matching)

        setting_layout.addStretch()
        # Add the settings widget to the main layout
        main_layout.addWidget(setting_widget)

    def do_matching(self):
        if self.method is None:
            self.update_method()
        if self.method.is_target_set() is False:
            self.method.set_target(self.map)
        T_new = self.method.align(self.scan, init_T=np.eye(4))
        self.scan = transform_points(T_new, self.scan)
        self['scan'].set_data(self.scan)

        msg_box = QDialog(self)
        msg_box.resize(600, 300)
        msg_box.setWindowTitle("Matching Finished")

        # Create a text edit widget to show the camera parameters
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)

        quat = q3d.matrix_to_quaternion(T_new[:3, :3])
        trans = T_new[:3, 3]
        # Format the camera parameters
        text = (
            f"Transformation Matrix (T):\n{T_new.round(4)}\n\n"
            f"Quaternion (x, y, z, w):\n{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}\n\n"
            f"Translation (x, y, z):\n{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}\n"
        )
        text_edit.setText(text)

        layout = q3d.QVBoxLayout()
        layout.addWidget(text_edit)
        msg_box.setLayout(layout)
        msg_box.exec()

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
        max_dist = self.box_max_dist.value()
        if index == 0:
            self.box_v.setHidden(True)
            self.box_k.setHidden(True)
            self.method = ICP(max_dist=max_dist)
            self['map'].set_color_mode('FLAT')
        elif index == 1:
            self.box_v.setHidden(True)
            self.box_k.setHidden(False)
            self['map'].set_color_mode('FLAT')
            k = self.box_k.value()
            self.method = PlaneICP(k=k, max_dist=max_dist)
        elif index == 2:
            self.box_k.setHidden(True)
            self.box_v.setHidden(False)
            voxel_size = self.box_v.value()
            self.method = NDT(voxel_size=voxel_size, max_dist=max_dist)
            self['map'].set_color_mode('RGB')
        elif index == 3:
            self.box_k.setHidden(True)
            self.box_v.setHidden(False)
            voxel_size = self.box_v.value()
            self.method = VPlaneICP(voxel_size=voxel_size, max_dist=max_dist)
            self['map'].set_color_mode('RGB')
        else:
            raise ValueError("Invalid method selected.")

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

    app = q3d.QApplication([])
    viewer = DEMOViewer(name='Demo Matching')
    viewer.show()
    app.exec()
