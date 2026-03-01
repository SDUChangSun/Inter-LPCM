from dataprepare.data import dataPrepare
import glob,datetime,os
import pt as pointCloud
import numpy as np 
import random
import matplotlib.pyplot as plt

def readdata(oriDir, sample_rate, seq,fangwei_step):
    train_data = []
    pt_elevation = np.empty((0,5))
    for folder in range(12,22):
        folder = '{:02d}'.format(folder)
        fileList = sorted(glob.glob(oriDir+folder+'/*.bin'))
        for oriFile in fileList:
            ptName = os.path.splitext(os.path.basename(oriFile))[0]
            p, refPt, split_p = dataPrepare(oriFile,fangwei_step,saveMatDir='./Data/testPly')
            recon_pt = np.empty((0,5))
            q = 0
            mean = 0
            for lidar in p:
                lidar2 = np.zeros((lidar.shape[0],5))
                lidar3 = np.zeros((lidar.shape[0],4))
                #draw_picture(lidar,ptName,q)
                mean = np.mean(lidar,axis=0)
                lidar2[:,0] = lidar[:,0]
                lidar2[:,1] = mean[1]
                lidar2[:,2] = lidar[:,2]
                lidar2[:,3] = q
                lidar2[:,4] = lidar[:,1]
                recon_pt = np.vstack((recon_pt, lidar2))
                lidar3[:, 3] = q 
                lidar3[:,0:3] = lidar 
                lidar3[:,1] = mean[1]
                if len(lidar) == 1: 
                    continue
                zero_rows = np.zeros((seq-1, 4))
                for j in range (seq-1):
                    zero_rows[j,:] = lidar3[0,:]-np.array([2,0,0,0])*(seq-1-j)
                #print(zero_rows)                 
                lidar3 = np.vstack((zero_rows, lidar3))
                for i in range(sample_rate):
                    max_len = len(lidar3) 
                    start_index = random.randint(0, max_len - seq) 
                    end_index = start_index + seq 
                    sub_list = lidar3[start_index:end_index] 
                    train_data.append(sub_list)
                q = q+1
            random_array = np.random.rand(len(recon_pt[:,2]))
            recon_pt[:,2] = np.round(recon_pt[:,2]*100+random_array)/100
            pt_elevation = np.vstack((pt_elevation, recon_pt))
            #psnr(refPt, recon_pt)
    train_data = np.array(train_data)
    return train_data,  pt_elevation

def save_list_to_txt(lst, filename): 
    with open(filename, 'wb') as f: 
        for item in lst: 
            np.set_printoptions(threshold=np.inf)
            f.write(str(item).encode() + b'\n')

def readbin(oriDir):
    train_data = []
    for folder in range(12,22):
        folder = '{:02d}'.format(folder)
        fileList = sorted(glob.glob(oriDir+folder+'/*.bin'))
        for oriFile in fileList:
            ptName = os.path.splitext(os.path.basename(oriFile))[0]
            p = dataPrepare(oriFile,saveMatDir='./Data/testPly')
        #print(oriDir+folder+'velodyne/*.bin')
    return p 

def spherical_to_cartesian(points,degree):
    azimuth = np.radians(points[:, 0])*0.045 
    elevation = np.radians(points[:, 1])
    radius = points[:, 2]
    if degree == True:
        azimuth = np.where(azimuth>=180,azimuth-360, azimuth) 
    x = radius * np.cos(azimuth) 
    y = radius * np.sin(azimuth)
    z = radius * np.tan(elevation) # 创建笛卡尔坐标数组 
    cartesian_coords = np.column_stack((x, y, z))
    return cartesian_coords

def psnr(refPt, p ):
    p = spherical_to_cartesian(p,degree = True)
    p = p - np.mean(p,axis=0)
    p = p/abs(p).max()
    refPt = refPt - np.mean(refPt,axis=0)
    refPt = refPt/abs(refPt).max()
    pointCloud.pcerror(refPt,p,None,'-r 1',None).wait()
def draw_picture(p,fileName,q):
    x = p[:, 2]  # 获取第三列  
    y = p[:, 1]  # 获取第二
    fig, ax = plt.subplots()  
    ax.scatter(x, y)  
    plt.savefig('./figure/'+fileName+str(q)+'.png', bbox_inches='tight') 
    plt.close()
if __name__=="__main__":
    
    p = dataPrepare('./test/12/000092.bin',saveMatDir='./Data/testPly')
    for lidar in p:
        lidar = sorted(lidar, key=lambda x: x[0])
        print(len(lidar))
    '''train_data = []
    for folder in range(1,2):
        folder = '{:02d}'.format(folder)
        fileList = sorted(glob.glob(oriDir+folder+'/velodyne/*.bin'))'''