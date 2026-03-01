import pt as pointCloud
import numpy as np
import os
import hdf5storage
import math
import sys
import matplotlib.pyplot as plt
import matplotlib
import deepCABAC
import open3d as o3d
from s_r2 import registion
from get_t import semantic


def get_split_num_reverse(split_p, threshold=0.4):
    """
    反向检测：从最后一个元素开始往前检测连续两个方差大于阈值的情况
    
    参数:
    split_p: 包含numpy数组的列表
    threshold: 阈值，默认为0.4
    
    返回:
    第一个满足条件的索引i（从后往前找的第一个），如果没有则返回-1
    """
    std_devs = []  # 标准差列表
    valid_indices = []  # 记录有效数组的索引
    
    # 计算所有数组第3列的标准差
    for i, arr in enumerate(split_p):
        if arr.shape[1] >= 3:  # 确保至少有3列
            third_col = arr[:, 2]  # 获取第3列（索引为2）
            std_dev = np.std(third_col)  # 计算标准差
            std_devs.append(std_dev)
            valid_indices.append(i)
    
    # 从后往前检测连续两个std_devs > threshold的情况
    for j in range(len(std_devs) - 1, 0, -1):  # 从最后一个到第二个
        if std_devs[j] > threshold and std_devs[j - 1] > threshold:
            # 返回原列表中的索引（注意：这里返回的是前一个索引）
            return valid_indices[j - 1]  # 返回第一个满足条件的索引（从后往前找）
    
    return -1  # 没有找到满足条件的情况


def dataPrepare(fileName,fileName2,fangwei_step,saveMatDir='Data',ptNamePrefix=''):
    #larsez = np.array([29.900000, 26.600000, 28.300000, 24.600000, 26.800000, 25.100000, 24.800000, 22.400000, 22.400000, 21.900000, 23.000000, 20.700000, 21.100000, 20.300000, 19.900000, 19.000000, 18.900000, 15.300000, 17.300000, 16.000000, 16.200000, 15.100000, 14.800000, 14.400000, 13.800000, 13.000000, 12.700000, 12.100000, 11.500000, 11.000000, 10.400000,  9.800000, 10.700000, 10.300000, 10.000000,  9.400000, 9.100000,  8.600000,  8.200000,  7.700000,  7.400000,  6.800000, 6.500000,  6.000000,  5.600000,  5.100000,  4.700000,  4.300000, 3.900000,  3.500000,  3.000000,  2.600000,  2.100000,  1.800000, 1.300000,  0.900000,  0.500000, -0.100000, -0.400000, -0.900000, -1.200000, -1.700000, -2.100000, -2.500000])
    #larsez = 0.001*larsez
    p = load_kitti_bin(fileName)[:, :3]
    ref_p = load_kitti_bin(fileName2)[:, :3]
    refPt = p
    split_p, p2 ,phi = construct_predtree(fileName,p ,fangwei_step,degree = True) 
    split_number = get_split_num_reverse(split_p)
    _,indices_ref_p = semantic(ref_p)
    ref_p_semantic = ref_p[:indices_ref_p[split_number],:]
    
    _,indices_p = semantic(p)
    p_semantic = p[:indices_p[split_number],:]

    ref_p_semantic = registion(ref_p_semantic[:, :3],fileName)
    inter_mode = get_inter_mode(ref_p_semantic,p_semantic)
    
    s_ref_p_semantic,_ = cartesian_to_spherical(ref_p_semantic ,fangwei_step,degree = True)
    
    split_ref_p_semantic = np.split(s_ref_p_semantic, indices_ref_p[:split_number])
    split_ref_p_semantic_withid = add_id(split_ref_p_semantic)
    
    return split_p,split_ref_p_semantic_withid, refPt, p2 ,inter_mode

def get_inter_mode(ref_p_semantic,p_semantic):
    
    tmp_test_file = "temp/pcerror_results.txt"
    pointCloud.pcerror(ref_p_semantic,p_semantic,None,'-r 50',tmp_test_file).wait()
    d1,_ = get_psnr(tmp_test_file)
    print('similar:',d1)
    if d1>35:
        mode = True
    else:
        mode = False    
    return mode

def get_psnr(tmp_test_file):
    with open(tmp_test_file) as f:
        c = f.readlines()
    f.close()
    for i in range(len(c)):
        if c[i].startswith('3.'):
            d1 = float(c[i+2].split(' ')[-1])
            try:
                d2 = float(c[i+4].split(' ')[-1])
            except Exception as e:
                d2 = 0.
            break
    os.remove(tmp_test_file)
    return d1, d2

def dataPrepare2(fileName,fangwei_step,saveMatDir='Data',ptNamePrefix=''):
    #larsez = np.array([29.900000, 26.600000, 28.300000, 24.600000, 26.800000, 25.100000, 24.800000, 22.400000, 22.400000, 21.900000, 23.000000, 20.700000, 21.100000, 20.300000, 19.900000, 19.000000, 18.900000, 15.300000, 17.300000, 16.000000, 16.200000, 15.100000, 14.800000, 14.400000, 13.800000, 13.000000, 12.700000, 12.100000, 11.500000, 11.000000, 10.400000,  9.800000, 10.700000, 10.300000, 10.000000,  9.400000, 9.100000,  8.600000,  8.200000,  7.700000,  7.400000,  6.800000, 6.500000,  6.000000,  5.600000,  5.100000,  4.700000,  4.300000, 3.900000,  3.500000,  3.000000,  2.600000,  2.100000,  1.800000, 1.300000,  0.900000,  0.500000, -0.100000, -0.400000, -0.900000, -1.200000, -1.700000, -2.100000, -2.500000])
    #larsez = 0.001*larsez
    p = load_kitti_bin(fileName)
    refPt = p[:, :3]
    split_p, p2 ,phi = construct_predtree(fileName,p[:, :3] ,fangwei_step,degree = True) 
    return split_p, refPt, p2 ,phi
    
def add_id(p):
    new_p = []  # 用于存储处理后的点云
    q = 0       # 初始标签值
    for lidar in p:
    # 创建带标签的4维点云 (N×4)
        lidar3 = np.zeros((lidar.shape[0], 4))
        lidar3[:, 0:3] = lidar  # 前三维是原始坐标
        lidar3[:, 3] = q        # 第四维是标签
        new_p.append(lidar3)    # 添加到新列表
        q += 1                  # 标签自增
    return new_p



def construct_predtree(fileName,p ,fangwei_step,degree = True):    
    p2,phi = cartesian_to_spherical(p ,fangwei_step,degree)
    skip = int((360/fangwei_step)*0.1)
    split_indices = np.where(p2[:-1,0] > p2[1:,0]+skip)[0]+1
    #if len(split_indices)!=63:
       #print(fileName+ '  split error:',len(split_indices)+1)
    '''group_ids = np.zeros_like(p[:, 0])  
    group_start = 0  
    for i, idx in enumerate(split_indices):  
        group_ids[group_start:idx-1] = larsez[i]  
        group_start = idx  
    group_ids[group_start:] = larsez[i+1] 
    p[:, 2] = p[:, 2]-group_ids
    p = cartesian_to_spherical(p ,fangwei_step,degree = True)
    draw_picture(p,fileName) '''
    split_p = np.split(p2, split_indices)
    return split_p, p2 ,phi
    
    
def load_kitti_bin(bin_file):
    """
    加载 KITTI 数据集的 .bin 文件
    :param bin_file: .bin 文件路径
    :return: numpy.ndarray, 形状为 (N, 4), 每行表示一个点 (x, y, z, intensity)
    """
    if bin_file == 'r_frame.bin':
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 3)
    else:
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points
    

def write_ply_file(filename, vertices, normal,reflectance_value=15):   
    vertices = np.asarray(vertices, dtype=np.float32)  
  
    # 固定的法线向量和反射率值  
    normal_vector = np.asarray(normal, dtype=np.float32) 
    reflectance = np.full(vertices.shape[0], reflectance_value, dtype=np.uint16)  
  
    # 打开文件以写入  
    with open(filename, 'w') as f:  
        # 写入PLY文件头  
        f.write("ply\n")  
        f.write("format ascii 1.0\n")  
        f.write(f"element vertex {vertices.shape[0]}\n")  
        f.write("property float32 x\n")  
        f.write("property float32 y\n")  
        f.write("property float32 z\n")  
        f.write("property uint16 reflectance\n")  
        f.write("property float32 nx\n")  
        f.write("property float32 ny\n")  
        f.write("property float32 nz\n")  
        f.write("element face 0\n")  
        f.write("property list uint8 int32 vertex_index\n")  
        f.write("end_header\n")  
  
        # 写入顶点数据  
        for i in range(vertices.shape[0]):    
            f.write(f"{vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {vertices[i, 2]:.6f} {reflectance[i]} "  
                    f"{normal_vector[i, 0]:.6f} {normal_vector[i, 1]:.6f} {normal_vector[i, 2]:.6f}\n")

def getnormal(p,m):
    # 将NumPy数组转换为Open3D点云对象  
    pcd = o3d.geometry.PointCloud()  
    pcd.points = o3d.utility.Vector3dVector(p)  
    # 法向量估计  
    # 使用KDTree进行邻域搜索，并计算法向量   
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1/m, max_nn=30))  
    # 如果你需要获取法向量数据（作为numpy数组）  
    normals = np.asarray(pcd.normals)  
    return normals

def psnr(refPt, p ,fangwei_step):
    #p[:, 1] = np.round(p[:, 1]*10)/10
    #p[:, 2] = np.round(p[:, 2])
    p = spherical_to_cartesian(p,fangwei_step,degree = True)
    p.tofile('r_frame.bin')
    res = refPt-p
    res_flatten = res.flatten()
    quantires = np.round(res_flatten*50)
    dequantires = quantires/50
    dequantires = dequantires.reshape(p.shape)
    p = p + dequantires
    encoder = deepCABAC.Encoder()
    encoder.encodeWeightsRD(quantires[0:50000].astype(np.float32), 1.0, 1.0, 0.0)
    encoder.encodeWeightsRD(quantires[50000:100000].astype(np.float32), 1.0, 1.0, 0.0)
    encoder.encodeWeightsRD(quantires[100000:150000].astype(np.float32), 1.0, 1.0, 0.0)
    encoder.encodeWeightsRD(quantires[150000:210000].astype(np.float32), 1.0, 1.0, 0.0)
    encoder.encodeWeightsRD(quantires[210000:270000].astype(np.float32), 1.0, 1.0, 0.0)
    encoder.encodeWeightsRD(quantires[270000:330000].astype(np.float32), 1.0, 1.0, 0.0)
    encoder.encodeWeightsRD(quantires[330000:].astype(np.float32), 1.0, 1.0, 0.0)
    stream = encoder.finish()
    #print(encode_res2[100:200])
    stream_bytes = stream.tobytes()
    bpip = len(stream_bytes)*8/len(quantires)
    '''p = p - np.mean(p,axis=0)
    p = p/abs(p).max() 
    refPt = refPt - np.mean(refPt,axis=0)'''
    m = abs(refPt).max()
    #refPt = refPt/abs(refPt).max()
    normal = getnormal(refPt,m)
    normal_name = 'normal.ply'
    write_ply_file(normal_name, refPt,normal)
    pointCloud.pcerror(refPt,p,normal_name,'-r 59.7',None).wait()    
    return bpip
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

def spherical_to_cartesian(points,fangwei_step,degree):
    azimuth = points[:, 0]*fangwei_step
    elevation = np.radians(points[:, 1])
    radius = points[:, 2]
    if degree == True:
        azimuth = np.where(azimuth>=180,azimuth-360, azimuth) 
        azimuth = np.radians(azimuth)
    x = radius * np.cos(azimuth) 
    y = radius * np.sin(azimuth)
    z = radius * np.tan(elevation) # 创建笛卡尔坐标数组 
    cartesian_coords = np.column_stack((x, y, z))
    return cartesian_coords

def save_list_to_txt(lst, filename): 
    with open(filename, 'wb') as f: 
        for item in lst: 
            np.set_printoptions(threshold=np.inf)
            f.write(str(item).encode() + b'\n')

def draw_picture(p,fileName):
    x = p[:, 2]  # 获取第三列  
    y = p[:, 1]  # 获取第二
    fig, ax = plt.subplots()  
    ax.scatter(x, y)  
    plt.savefig('line_plot.png', bbox_inches='tight') 

