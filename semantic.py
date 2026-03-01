import open3d as o3d
import numpy as np

def segment_ground_ransac(pcd, distance_threshold=0.3, ransac_n=10, num_iterations=100000):
    """
    使用 RANSAC 进行地面分割
    :param pcd: 输入点云 (Open3D PointCloud)
    :param distance_threshold: RANSAC 平面拟合阈值（控制哪些点属于地面）
    :param ransac_n: 参与 RANSAC 拟合的最少点数
    :param num_iterations: RANSAC 迭代次数
    :return: ground_pcd (地面点云), non_ground_pcd (非地面点云)
    """
    points = np.asarray(pcd.points)
    indices = np.arange(len(points)) 
    # RANSAC 拟合平面
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    
    # 获取地面点
    inliers = np.array(inliers)
    outliers = np.setdiff1d(indices, inliers)

    return inliers, outliers

def visualize_colored_point_cloud(ground_pcd, non_ground_pcd):
    """
    可视化地面 & 非地面点云（绿色 = 地面，红色 = 非地面）
    """
    ground_pcd.paint_uniform_color([0, 1, 0])  # 绿色 (Ground)
    non_ground_pcd.paint_uniform_color([1, 0, 0])  # 红色 (Non-Ground)

    o3d.visualization.draw_geometries([ground_pcd, non_ground_pcd], window_name="RANSAC 地面分割")
    
def numpy_to_open3d(points):
    """
    将 numpy 数组转换为 open3d 点云对象
    :param points: numpy.ndarray, 形状为 (N, 3) 或 (N, 4)
    :return: open3d.geometry.PointCloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 只取 x, y, z
    return pcd

if __name__ == "__main__":
    # 读取点云文件（替换为你的点云数据）
    source_path = "000000.bin"  # 源点云路径
    points = np.fromfile(source_path, dtype=np.float32).reshape(-1, 4)
    vel_msg = points[:, :3]
    vel_msg = vel_msg * np.array([1,1,-1])
    # 转换为 open3d 点云对象
    pcd = numpy_to_open3d(vel_msg)

    # 1. 进行 RANSAC 地面分割
    ground_pcd, non_ground_pcd = segment_ground_ransac(pcd)

    # 2. 可视化带颜色的点云
    visualize_colored_point_cloud(ground_pcd, non_ground_pcd)

    # 3. 保存分割结果
    o3d.io.write_point_cloud("ground.ply", ground_pcd)
    o3d.io.write_point_cloud("non_ground.ply", non_ground_pcd)
