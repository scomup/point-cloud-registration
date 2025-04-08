#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/random_sample.h>

#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <iostream>

Eigen::Matrix3f expSO3(const Eigen::Vector3f& w) {
  // Compute the exponential map for SO(3) (rotation matrix)
  float theta = w.norm();
  if (theta < 1e-6) {
    return Eigen::Matrix3f::Identity();
  }
  Eigen::Matrix3f W;
  W << 0, -w.z(), w.y(), w.z(), 0, -w.x(), -w.y(), w.x(), 0;
  return Eigen::Matrix3f::Identity() + (std::sin(theta) / theta) * W +
         ((1 - std::cos(theta)) / (theta * theta)) * (W * W);
}

std::pair<pcl::PointCloud<pcl::PointNormal>::Ptr, double> test_pcl_estimate_normals(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int k) {
  // Compute normals for the input cloud and measure execution time
  auto start = std::chrono::high_resolution_clock::now();

  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(
      new pcl::PointCloud<pcl::PointNormal>());
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
      new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());

  ne.setInputCloud(cloud);
  ne.setSearchMethod(tree);
  ne.setKSearch(k);  // Use k nearest neighbors
  ne.compute(*normals);

  // Combine XYZ and normals into PointNormal
  pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count() /
      1000.0;

  return {cloud_with_normals, elapsed_time};
}

void test_icp(const pcl::PointCloud<pcl::PointXYZ>::Ptr& map_cloud,
              const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud,
              int max_iter, double tol, double max_dist, double& elapsed_time) {
  auto start = std::chrono::high_resolution_clock::now();
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(scan_cloud);
  icp.setInputTarget(map_cloud);
  // Use passed parameters
  icp.setMaximumIterations(max_iter);
  icp.setTransformationEpsilon(tol);
  icp.setMaxCorrespondenceDistance(max_dist);
  pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
  icp.align(aligned_cloud);
  auto end = std::chrono::high_resolution_clock::now();
  elapsed_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count() /
      1000.0;
  if (icp.hasConverged()) {
    std::cout << "Point-to-Point ICP converged. Fitness score: "
              << icp.getFitnessScore() << std::endl;
  } else {
    std::cerr << "Point-to-Point ICP did not converge." << std::endl;
  }
}

void test_point_to_plane_icp(
    const pcl::PointCloud<pcl::PointNormal>::Ptr& map_cloud_with_normals,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud, int max_iter,
    double tol, double max_dist, double& elapsed_time) {
      
      // Convert scan cloud to PointNormal (normals are not used for the source
      // cloud)
      pcl::PointCloud<pcl::PointNormal>::Ptr scan_cloud_with_normals(
        new pcl::PointCloud<pcl::PointNormal>());
        pcl::copyPointCloud(*scan_cloud, *scan_cloud_with_normals);
        
  auto start = std::chrono::high_resolution_clock::now();
  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
  icp.setInputSource(scan_cloud_with_normals);
  icp.setInputTarget(map_cloud_with_normals);

  // Set parameters using passed arguments
  icp.setMaximumIterations(max_iter);
  icp.setTransformationEpsilon(tol);
  icp.setMaxCorrespondenceDistance(max_dist);

  pcl::PointCloud<pcl::PointNormal> aligned_cloud;
  icp.align(aligned_cloud);

  auto end = std::chrono::high_resolution_clock::now();
  elapsed_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count() /
      1000.0;

  if (icp.hasConverged()) {
    std::cout << "Point-to-Plane ICP converged. Fitness score: "
              << icp.getFitnessScore() << std::endl;
  } else {
    std::cerr << "Point-to-Plane ICP did not converge." << std::endl;
  }
}

void test_ndt(const pcl::PointCloud<pcl::PointXYZ>::Ptr& map_cloud,
              const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud,
              int max_iter, double tol, float voxel_size,
              double& elapsed_time) {
  auto start = std::chrono::high_resolution_clock::now();

  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  ndt.setInputSource(scan_cloud);
  ndt.setInputTarget(map_cloud);

  // Set parameters using passed arguments
  ndt.setMaximumIterations(max_iter);
  ndt.setTransformationEpsilon(tol);
  ndt.setResolution(voxel_size);

  pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
  ndt.align(aligned_cloud);

  auto end = std::chrono::high_resolution_clock::now();
  elapsed_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count() /
      1000.0;

  if (ndt.hasConverged()) {
    std::cout << "NDT converged. Fitness score: " << ndt.getFitnessScore()
              << std::endl;
  } else {
    std::cerr << "NDT did not converge." << std::endl;
  }
}

double test_pcl_voxel_filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                             float voxel_size) {
  std::cout << "\nTesting PCL Voxel Filter..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(cloud);
  voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  voxel_filter.filter(*filtered_cloud);

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count() /
      1000.0;
  std::cout << "PCL Voxel Filter execution time: " << elapsed_time << " s"
            << std::endl;

  return elapsed_time;
}

int main() {
  // File path for the map point cloud
  std::string map_file = "../../data/B-01.pcd";

  // Load map point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_file, *map_cloud) == -1) {
    std::cerr << "Failed to load map point cloud: " << map_file << std::endl;
    return -1;
  }

  // Apply rigid transformation to generate the scan point cloud
  Eigen::Matrix3f R =
      expSO3(Eigen::Vector3f(0.0, 0.0, 0.0));  // Identity rotation
  Eigen::Vector3f t(0.0, 0.0, 0.3);            // Translation vector

  pcl::PointCloud<pcl::PointXYZ>::Ptr scan_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  for (const auto& point : map_cloud->points) {
    Eigen::Vector3f p(point.x, point.y, point.z);
    Eigen::Vector3f transformed_p = R * p + t;
    scan_cloud->points.emplace_back(transformed_p.x(), transformed_p.y(),
                                    transformed_p.z());
  }
  /*
      num_points = 100000
    indices = np.random.choice(scan.shape[0], num_points, replace=False)
    scan = scan[indices]

*/
  scan_cloud->width = scan_cloud->points.size();
  scan_cloud->height = 1;
  scan_cloud->is_dense = true;

  // Randomly sample the scan cloud
  int num_points = 100000;
  pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_scan_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  pcl::RandomSample<pcl::PointXYZ> random_sample;
  random_sample.setInputCloud(scan_cloud);
  random_sample.setSample(num_points);
  random_sample.filter(*sampled_scan_cloud);

  // Replace scan_cloud with sampled_scan_cloud for further processing
  scan_cloud = sampled_scan_cloud;

  // Parameters (consistent with Python implementation)
  int k = 5;                // Number of nearest neighbors for normal estimation
  float voxel_size = 1.f;     // Voxel size
  int max_iter = 30;        // Maximum iterations for ICP and NDT
  double tol = 1e-3;        // Tolerance for convergence
  double max_dist =voxel_size * 2;    // Maximum correspondence distance

  // Compute normals for the map cloud and measure time
  auto result =
      test_pcl_estimate_normals(map_cloud, k);
  pcl::PointCloud<pcl::PointNormal>::Ptr map_cloud_with_normals = result.first;
  double pcl_normal_time = result.second;


  // Test PCL Voxel Filter
  double pcl_voxel_time = test_pcl_voxel_filter(map_cloud, voxel_size);

  // Test NDT
  double ndt_time;
  test_ndt(map_cloud, scan_cloud, max_iter, tol, voxel_size, ndt_time);

  // Test Point-to-Point ICP
  double icp_time;
  test_icp(map_cloud, scan_cloud, max_iter, tol, max_dist, icp_time);

  // Test Point-to-Plane ICP
  double plane_icp_time;
  test_point_to_plane_icp(map_cloud_with_normals, scan_cloud, max_iter, tol,
                          max_dist, plane_icp_time);

  // Output comparison table
  std::cout << "\nSpeed Comparison Table:\n";
  std::cout << std::left << std::setw(35) << "Algorithm" << std::right
            << std::setw(20) << "Execution Time (s)\n";
  std::cout << std::string(55, '-') << "\n";
  std::cout << std::left << std::setw(35) << "PCL Normal Estimation"
            << std::right << std::setw(20) << pcl_normal_time << "\n";
  std::cout << std::left << std::setw(35) << "PCL Voxel Filter" << std::right
            << std::setw(20) << pcl_voxel_time << "\n";
  std::cout << std::left << std::setw(35) << "NDT" << std::right
            << std::setw(20) << ndt_time << "\n";
  std::cout << std::left << std::setw(35) << "Point-to-Point ICP" << std::right
            << std::setw(20) << icp_time << "\n";
  std::cout << std::left << std::setw(35) << "Point-to-Plane ICP" << std::right
            << std::setw(20) << plane_icp_time << "\n";

  return 0;
}
