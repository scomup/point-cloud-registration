
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/point-cloud-registration.svg)](https://badge.fury.io/py/point-cloud-registration)

`point-cloud-registration` is a pure Python implementation of a point cloud registration library.


## Installation

To install `point-cloud-registration`, execute the following command in your terminal:

```bash
pip install point-cloud-registration
```

## Using

```python
#!/usr/bin/env python3

from point_cloud_registration import VoxelPoint2PlaneICP

icp = VoxelPoint2PlaneICP(voxel_size=0.5, max_iter=100, max_dist=2, tol=1e-3)
icp.update_target(target) # target is a Nx3 point numpy array
T_new = icp.fit(scan, init_T=np.eye(4)) # scan is a Mx3 numpy array

from point_cloud_registration.math_tools import makeRt
R_new, t_new = makeRt(T_new)
```

Enjoy it!