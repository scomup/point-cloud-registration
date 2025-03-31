#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Dense>
#include <iostream>

Eigen::Matrix3f expSO3(const Eigen::Vector3f& w) {
    // Compute the exponential map for SO(3) (rotation matrix)
    float theta = w.norm();
    if (theta < 1e-6) {
        return Eigen::Matrix3f::Identity();
    }
    Eigen::Matrix3f W;
    W << 0, -w.z(), w.y(),
         w.z(), 0, -w.x(),
        -w.y(), w.x(), 0;
    return Eigen::Matrix3f::Identity() + (std::sin(theta) / theta) * W + ((1 - std::cos(theta)) / (theta * theta)) * (W * W);
}

pcl::PointCloud<pcl::PointNormal>::Ptr compute_normals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // Compute normals for the input cloud
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());

    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(15);  // Use 15 nearest neighbors
    ne.compute(*normals);

    // Combine XYZ and normals into PointNormal
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
    return cloud_with_normals;
}

void test_icp(const pcl::PointCloud<pcl::PointXYZ>::Ptr& map_cloud,
              const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud) {
    std::cout << "\nTesting Point-to-Point ICP..." << std::endl;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(scan_cloud);
    icp.setInputTarget(map_cloud);

    // Set parameters
    icp.setMaximumIterations(30);  // max_iter
    icp.setTransformationEpsilon(1e-3);  // tol
    icp.setMaxCorrespondenceDistance(2.0);  // max_dist

    pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
    icp.align(aligned_cloud);

    if (icp.hasConverged()) {
        std::cout << "Point-to-Point ICP converged. Fitness score: " << icp.getFitnessScore() << std::endl;
        std::cout << "Transformation matrix:" << std::endl;
        std::cout << icp.getFinalTransformation() << std::endl;
    } else {
        std::cerr << "Point-to-Point ICP did not converge." << std::endl;
    }
}

void test_point_to_plane_icp(const pcl::PointCloud<pcl::PointNormal>::Ptr& map_cloud_with_normals,
                             const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud) {
    std::cout << "\nTesting Point-to-Plane ICP..." << std::endl;

    // Convert scan cloud to PointNormal (normals are not used for the source cloud)
    pcl::PointCloud<pcl::PointNormal>::Ptr scan_cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
    pcl::copyPointCloud(*scan_cloud, *scan_cloud_with_normals);

    pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
    icp.setInputSource(scan_cloud_with_normals);
    icp.setInputTarget(map_cloud_with_normals);

    // Set parameters
    icp.setMaximumIterations(30);  // max_iter
    icp.setTransformationEpsilon(1e-3);  // tol
    icp.setMaxCorrespondenceDistance(2.0);  // max_dist

    pcl::PointCloud<pcl::PointNormal> aligned_cloud;
    icp.align(aligned_cloud);

    if (icp.hasConverged()) {
        std::cout << "Point-to-Plane ICP converged. Fitness score: " << icp.getFitnessScore() << std::endl;
        std::cout << "Transformation matrix:" << std::endl;
        std::cout << icp.getFinalTransformation() << std::endl;
    } else {
        std::cerr << "Point-to-Plane ICP did not converge." << std::endl;
    }
}

void test_ndt(const pcl::PointCloud<pcl::PointXYZ>::Ptr& map_cloud,
              const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud) {
    std::cout << "\nTesting NDT..." << std::endl;

    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setInputSource(scan_cloud);
    ndt.setInputTarget(map_cloud);

    // Set parameters
    ndt.setMaximumIterations(30);  // max_iter
    ndt.setTransformationEpsilon(1e-3);  // tol
    ndt.setStepSize(0.1);  // Step size for NDT optimization
    ndt.setResolution(0.5);  // Resolution of the NDT grid

    pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
    ndt.align(aligned_cloud);

    if (ndt.hasConverged()) {
        std::cout << "NDT converged. Fitness score: " << ndt.getFitnessScore() << std::endl;
        std::cout << "Transformation matrix:" << std::endl;
        std::cout << ndt.getFinalTransformation() << std::endl;
    } else {
        std::cerr << "NDT did not converge." << std::endl;
    }
}

int main() {
    // File path for the map point cloud
    std::string map_file = "/home/liu/tmp/recorded_frames/clouds/0.pcd";

    // Load map point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_file, *map_cloud) == -1) {
        std::cerr << "Failed to load map point cloud: " << map_file << std::endl;
        return -1;
    }

    // Apply rigid transformation to generate the scan point cloud
    Eigen::Matrix3f R = expSO3(Eigen::Vector3f(0.0, 0.0, 0.0));  // Identity rotation
    Eigen::Vector3f t(0.3, 0.3, 0.3);  // Translation vector

    pcl::PointCloud<pcl::PointXYZ>::Ptr scan_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto& point : map_cloud->points) {
        Eigen::Vector3f p(point.x, point.y, point.z);
        Eigen::Vector3f transformed_p = R * p + t;
        scan_cloud->points.emplace_back(transformed_p.x(), transformed_p.y(), transformed_p.z());
    }
    scan_cloud->width = scan_cloud->points.size();
    scan_cloud->height = 1;
    scan_cloud->is_dense = true;

    // Compute normals for the map cloud
    pcl::PointCloud<pcl::PointNormal>::Ptr map_cloud_with_normals = compute_normals(map_cloud);
    auto start = std::chrono::high_resolution_clock::now();

    // Test NDT
    test_ndt(map_cloud, scan_cloud);

    auto ndt_end = std::chrono::high_resolution_clock::now();
    std::cout << "NDT execution time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(ndt_end - start).count() 
              << " ms" << std::endl;

    // Test Point-to-Point ICP
    auto icp_start = std::chrono::high_resolution_clock::now();
    test_icp(map_cloud, scan_cloud);

    auto icp_end = std::chrono::high_resolution_clock::now();
    std::cout << "Point-to-Point ICP execution time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(icp_end - icp_start).count() 
              << " ms" << std::endl;

    // Test Point-to-Plane ICP
    auto plane_icp_start = std::chrono::high_resolution_clock::now();
    test_point_to_plane_icp(map_cloud_with_normals, scan_cloud);

    auto plane_icp_end = std::chrono::high_resolution_clock::now();
    std::cout << "Point-to-Plane ICP execution time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(plane_icp_end - plane_icp_start).count() 
              << " ms" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total execution time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms" << std::endl;

    return 0;
}
