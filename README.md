Here’s your updated README with the **Roadmap** including GICP, Point-to-Point ICP, Point-to-Plane ICP, and NDT:  

---

# Point Cloud Registration  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![PyPI version](https://badge.fury.io/py/point-cloud-registration.svg)](https://badge.fury.io/py/point-cloud-registration)  

**`point-cloud-registration`** is a **pure Python**, lightweight, and fast point cloud registration library.  
It is designed to be **faster** than Open3D’s registration while using **only NumPy** for computations.  

## Features  
✅ **Pure Python** – No compiled extensions, works everywhere  
✅ **Fast & Lightweight** – Optimized algorithms with minimal overhead  
✅ **NumPy-based API** – Seamless integration with scientific computing workflows  

## Installation  

Install via pip:  

```bash
pip install point-cloud-registration
```

## Usage  

```python
#!/usr/bin/env python3

import numpy as np
from point_cloud_registration import VoxelPoint2PlaneICP

# Example point clouds
target = np.random.rand(100, 3)  # Nx3 point numpy array
scan = np.random.rand(80, 3)    # Mx3 point numpy array

icp = VoxelPoint2PlaneICP(voxel_size=0.5, max_iter=100, max_dist=2, tol=1e-3)
icp.update_target(target)  # Set the target point cloud
T_new = icp.fit(scan, init_T=np.eye(4))  # Fit the scan to the target
from point_cloud_registration.math_tools import makeRt
print("Estimated Transform matrix:\n", T_new)
```

## Roadmap  
🚀 **Upcoming Features & Enhancements**:  
- [ ] **Point-to-Point ICP** – Basic ICP implementation  
- [ ] **Point-to-Plane ICP** – Improved accuracy using normal constraints  
- [ ] **Generalized ICP (GICP)** – Handles anisotropic noise and improves robustness  
- [ ] **Normal Distributions Transform (NDT)** – Grid-based registration for high-noise environments  
- [ ] **Further optimizations** while staying pure Python  

## License  

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  

---

This version makes the **Roadmap** more structured and exciting! Let me know if you want any refinements. 🚀🔥