import numpy as np
import open3d as o3d
import pt as pointCloud
import os
import glob,datetime,os
import pt as pointCloud 
import random
#from rlconding import run_length_encode
import deepCABAC
from s_r import registion


def load_kitti_bin(bin_file):
    """
    加载 KITTI 数据集的 .bin 文件
    :param bin_file: .bin 文件路径
    :return: numpy.ndarray, 形状为 (N, 4), 每行表示一个点 (x, y, z, intensity)
    """
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points



def save_numpy(input_file,output_root, transform_matrix):
        
    parts = input_file.split(os.sep)
        # 假设路径结构为 ./xx/yy/cc/*.bin
    yy = parts[2]  # 第2部分是 yy
    cc = parts[3]  # 第3部分是 cc
    filename = parts[-1]  # 文件名（如 000001.bin）
        
        # 3. 构建输出路径（替换 xx 为 outdir）
    output_path = os.path.join(output_root, yy, cc, filename)
        
        # 4. 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 5. 保存矩阵为二进制文件
    transform_matrix.astype(np.float32).tofile(output_path)
    
    data = np.fromfile(output_path, dtype=np.float32)

def cartesian_to_spherical(points,fangwei_step,degree):
    spherical_points = []
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    r = np.sqrt(np.square(x)+np.square(y)+np.square(z))
    r1 = np.sqrt(np.square(x)+np.square(y))
    theta = np.arcsin(z/r)
    phi = np.arctan2(y, x)
    #print(phi,theta,)
    if degree == True:
        phi = np.rad2deg(phi)
        phi = np.where(phi<0, phi+360, phi)
        phi1 = np.round(phi/fangwei_step)
        theta = np.rad2deg(theta)
    else:
        phi = np.where(phi<0, phi+2*math.pi, phi)
    spherical_points = np.column_stack((phi1, theta, r1))
    return spherical_points,phi

def semantic(p):
    p2,phi = cartesian_to_spherical(p ,0.045,degree=True)
    skip = int((360/0.045)*0.1)
    split_indices = np.where(p2[:-1,0] > p2[1:,0]+skip)[0]+1
    split_p = np.split(p2, split_indices)
    return split_p, split_indices

