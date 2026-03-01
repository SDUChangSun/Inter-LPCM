import numpy as np
import open3d as o3d
import pt as pointCloud
from semantic import segment_ground_ransac
def icp_registration(source, target, max_iterations=50, tolerance=1e-6):
    """
    ICP 点云配准算法
    :param source: 源点云 (Open3D PointCloud)
    :param target: 目标点云 (Open3D PointCloud)
    :param max_iterations: 最大迭代次数
    :param tolerance: 收敛阈值
    :return: 变换矩阵 (4x4 numpy array), 配准后的点云
    """
    # 初始化变换矩阵（单位矩阵）
    transformation = np.identity(4)
    
    # 将点云转换为 numpy 数组
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    
    # 构建目标点云的 KDTree 加速最近邻搜索
    target_kd_tree = o3d.geometry.KDTreeFlann(target)
    
    prev_error = 0
    for i in range(max_iterations):
        # 1. 寻找最近邻对应点
        correspondences = []
        for src_point in source_points:
            _, idx, _ = target_kd_tree.search_knn_vector_3d(src_point, 1)
            correspondences.append(target_points[idx[0]])
        correspondences = np.array(correspondences)
        
        # 2. 计算点云质心
        src_centroid = np.mean(source_points, axis=0)
        tgt_centroid = np.mean(correspondences, axis=0)
        
        # 3. 去质心化
        src_centered = source_points - src_centroid
        tgt_centered = correspondences - tgt_centroid
        
        # 4. 计算旋转矩阵（SVD 分解）
        H = np.dot(src_centered.T, tgt_centered)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        # 确保旋转矩阵行列式为 1（避免反射）
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # 5. 计算平移向量
        t = tgt_centroid - np.dot(R, src_centroid)
        
        # 6. 构建变换矩阵
        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        # 更新累积变换
        transformation = np.dot(T, transformation)
        
        # 应用当前变换到源点云
        source_points = np.dot(source_points, R.T) + t
        
        # 计算误差（均方误差）
        error = np.mean(np.linalg.norm(source_points - correspondences, axis=1))
        
        # 检查是否收敛
        if np.abs(prev_error - error) < tolerance:
            break
        prev_error = error
    # 生成配准后的点云
    registered_pcd = o3d.geometry.PointCloud()
    registered_pcd.points = o3d.utility.Vector3dVector(source_points)
    return transformation, registered_pcd
    
def load_kitti_bin(bin_file):
    """
    加载 KITTI 数据集的 .bin 文件
    :param bin_file: .bin 文件路径
    :return: numpy.ndarray, 形状为 (N, 4), 每行表示一个点 (x, y, z, intensity)
    """
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    vel_msg = points[:, :3]
    vel_msg = vel_msg * np.array([1,1,-1]) 
    return vel_msg

def numpy_to_open3d(points):
    """
    将 numpy 数组转换为 open3d 点云对象
    :param points: numpy.ndarray, 形状为 (N, 3) 或 (N, 4)
    :return: open3d.geometry.PointCloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 只取 x, y, z
    return pcd


def registion(p,ref_p):
    # === 替换这里：加载您的点云文件 ===
    source_bin = ref_p
    target_bin = p
    source_point = numpy_to_open3d(source_bin)
    target_point = numpy_to_open3d(target_bin)

    # 运行 ICP 配准
    
    transformation_ground, registered_pcd_ground = icp_registration(source_point, target_point)


    return transformation_ground

