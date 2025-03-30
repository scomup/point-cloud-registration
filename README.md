# Point Cloud Registration  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/point-cloud-registration.svg?cache=1)](https://pypi.org/project/point-cloud-registration/)

**`point-cloud-registration`** is a **pure Python**, **lightweight**, and **fast** point cloud registration library.  
It outperforms Open3D’s registration in speed while relying **only on NumPy** for computations.

## Features  
✅ **Pure Python** – No compiled extensions, works everywhere  
✅ **Fast & Lightweight** – Optimized algorithms with minimal overhead  
✅ **NumPy-based API** – Seamless integration with scientific computing workflows  

The following registration algorithms are planned to be supported:

### Comparison of Registration Methods

| Method                        | Objective Function*                                         | Data Representation   | Speed         | Precision    |
|-------------------------------|-----------------------------------------------------------|------------------------|---------------|--------------|
| Point-to-Point ICP         | $\sum \| T p_i - q_i \|^2$   | Point-Based            | Fast          | Moderate     |
| Point-to-Plane ICP         | $\sum \| n_i^T (T p_i - q_i) \|^2 $ | Point-Based (with normals) | Fast | High | 
| Voxelized Point-to-Plane ICP         | $\sum \| n_i^T (T p_i - q_i) \|^2 $ | Voxel-Based (with normals) | Very Fast | High | 
| Generalized ICP (GICP)     | $\sum (T p_i - q_i)^T (C_i^Q + R C_i^P R^T)^{-1} (T p_i - q_i)$ | Point-Based (with covariances) | Moderate | Very High | 
| Normal Distributions Transform (NDT) | $\sum (T p_i - \mu_i)^T \Sigma_i^{-1} (T p_i - \mu_i)$ | Voxel-Based (with covariances) | Very Fast | Moderate |

---

### Speed Comparison Table

| Method                        | Points Size | Our Implementation (Time in seconds) | Open3D (Time in seconds) |
|-------------------------------|--------------|---------------------------------------|---------------------------|
| Point-to-Point ICP            | 400,000       | 16                                 | 61                     |
| Point-to-Plane ICP            | 400,000       | TODO                                 | TODO                     |
| Voxelized Point-to-Plane ICP  | 400,000       | 2.9                                 | N/A                      |
| Normal Distributions Transform (NDT) | 400,000 | 10                                 | N/A                     |

---

**Note**: The above times are based on synthetic datasets and may vary depending on hardware and dataset characteristics.

**For more details, check the documentation.*


## Installation  

Install via pip:  

```bash
pip install point-cloud-registration
pip install q3dviewer # (optional) for visual demo
pip install pykdtree # (optional) for fast kdtree
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
print("Estimated Transform matrix:\n", T_new)
```

## Roadmap  
🚀 **Upcoming Features & Enhancements**:  
- [x] **Point-to-Point ICP** – Basic ICP implementation  
- [ ] **Point-to-Plane ICP** – Improved accuracy using normal constraints  
- [ ] **Generalized ICP (GICP)** – Handles anisotropic noise and improves robustness  
- [x] **Normal Distributions Transform (NDT)** – Grid-based registration for high-noise environments  
- [ ] **Further optimizations** while staying pure Python  

## License  

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
