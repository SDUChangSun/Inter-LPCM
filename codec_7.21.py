import os
import numpy as np
import deepCABAC
import matplotlib.pyplot as plt
import pandas as pd

# read & write csv -----------------------------------------------------------------------------------------------------
def read_csv(dir):
    '''读取.csv文件'''
    data = np.loadtxt(dir, delimiter=',')
    return data

def write_csv(data, dir, filename):
    df_orig = pd.DataFrame(data.astype(np.float64))
    if not os.path.exists(dir): os.makedirs(dir)
    df_orig.to_csv(dir + os.path.splitext(filename)[0] + '.csv', index=False, header=False)
# ----------------------------------------------------------------------------------------------------------------------


# liear quantization & dequantization ----------------------------------------------------------------------------------
def trans_bit_linear(data, bit):
    '''数据 N bit线性量化'''  
    min_val = data.min().min()
    max_val = data.max().min()
    data_norm = (np.power(2, bit) - 1) * (data - min_val) / (max_val - min_val)  
    return np.round(data_norm), data.min().min(), data.max().max()


def inverse_trans_bit_linear(data_norm, bit, min_orig, max_orig):
    '''数据 N bit log反量化'''
    data_inverse = (((max_orig - min_orig) * data_norm) / (np.power(2, bit) - 1)) + min_orig
    
    return data_inverse
# ----------------------------------------------------------------------------------------------------------------------


# differential encoding & decoding--------------------------------------------------------------------------------------
def diff_enc(data):
    '''将输入的一行数据差分编码'''
    encoded_data = [data[0]]
    for i in range(1, len(data)):
        encoded_data.append(data[i] - data[i-1])
    return np.array(encoded_data)


def diff_dec(res):
    '''将输入的一行数据差分解码'''
    decoded_data = [res[0]]  
    for i in range(1, len(res)):
        decoded_data.append(decoded_data[i-1] + res[i])
    return np.array(decoded_data)
# ----------------------------------------------------------------------------------------------------------------------


# cabac encoding & decoding --------------------------------------------------------------------------------------------
def cabac_enc(data, step_size, dir):
    '''将输入数据利用cabac编码为码流'''
    encoder = deepCABAC.Encoder()
    encoder.encodeWeightsRD(data.astype(np.float32), 1.0 , step_size, 0.0)
    stream = encoder.finish()
    stream_bytes = stream.tobytes()
    with open(dir + '.bin', 'wb') as f_out:
        f_out.write(stream_bytes)
        
        
def cabac_dec(dir):
    '''将码流利用cabac解码为重建数据'''
    with open(dir + '.bin', 'rb') as fin:
        stream = np.frombuffer(fin.read(), dtype=np.uint8)
    decoder = deepCABAC.Decoder()
    decoder.getStream(stream)
    data = decoder.decodeWeights()
    return data
# ----------------------------------------------------------------------------------------------------------------------


# Prediction -----------------------------------------------------------------------------------------------------------

# RGB_Vertical
def rgb_ver_enc(cu_cur, line_up):
    '''Vertical prediction'''
    res = np.zeros(cu_cur.shape)
    if line_up[0] == -1:
        res[0, :] = diff_enc(cu_cur[0, :])
        for i in range(1, cu_cur.shape[0]):
            res[i, :] = cu_cur[i, :] - cu_cur[i-1, :]
        return res
    else:
        res[0, :] = cu_cur[0, :] - line_up
        for i in range(1, cu_cur.shape[0]):
            res[i, :] = cu_cur[i, :] - cu_cur[i-1, :]
        return res
    
def rgb_ver_dec(res, line_up):
    '''Vertical prediction'''
    cu_cur = np.zeros(res.shape)
    if line_up[0] == -1:
        cu_cur[0, :] = diff_dec(res[0, :])
        for i in range(1, res.shape[0]):
            cu_cur[i, :] = res[i, :] + cu_cur[i-1, :]
        return cu_cur
    else:
        cu_cur[0, :] = res[0, :] + line_up
        for i in range(1, res.shape[0]):
            cu_cur[i, :] = res[i, :] + cu_cur[i-1, :]
        return cu_cur

# RGB_Horizon
def rgb_hor_enc(cu_cur, line_left):
    '''Horizon prediction'''
    res = np.zeros(cu_cur.shape)
    if line_left[0] == -1:
        res[:, 0] = diff_enc(cu_cur[:, 0])
        for i in range(1, cu_cur.shape[1]):
            res[:, i] = cu_cur[:, i] - cu_cur[:, i-1]
        return res
    else:
        res[:, 0] = cu_cur[:, 0] - line_left
        for i in range(1, cu_cur.shape[1]):
            res[:, i] = cu_cur[:, i] - cu_cur[:, i-1]
        return res

def rgb_hor_dec(res, line_left):
    '''Vertical prediction'''
    cu_cur = np.zeros(res.shape)
    if line_left[0] == -1:
        cu_cur[:, 0] = diff_dec(res[:, 0])
        for i in range(1, res.shape[1]):
            cu_cur[:, i] = res[:, i] + cu_cur[:, i-1]
        return cu_cur
    else:
        cu_cur[:, 0] = res[:, 0] + line_left
        for i in range(1, res.shape[1]):
            cu_cur[:, i] = res[:, i] + cu_cur[:, i-1]
        return cu_cur

# RGB_Vertical & Horizon
def rgb_ver_hor_enc(cu_cur, line_up, line_left):
    '''Vertical & Horizon prediction'''
    res = np.zeros(cu_cur.shape)
    if (line_left[0] == -1) and (line_up[0] == -1):
        res[0, :] = diff_enc(cu_cur[0, :])
        res[:, 0] = diff_enc(cu_cur[:, 0])
        for i in range(1, cu_cur.shape[0]):
            for j in range(1, cu_cur.shape[1]):
                res[i, j] = cu_cur[i, j] - int((cu_cur[i-1, j] + cu_cur[i, j-1]) / 2)
        return res
    
    elif (line_left[0] == -1) and (line_up[0] != -1):
        res[:, 0] = diff_enc(cu_cur[:, 0])
        for i in range(0, cu_cur.shape[0]):
            for j in range(1, cu_cur.shape[1]):
                if i == 0:
                    res[i, j] = cu_cur[i, j] - int((cu_cur[i, j-1] + line_up[j]) / 2)
                else:
                    res[i, j] = cu_cur[i, j] - int((cu_cur[i-1, j] + cu_cur[i, j-1]) / 2)
        return res
    
    elif (line_left[0] != -1) and (line_up[0] == -1):
        res[0, :] = diff_enc(cu_cur[0, :])
        for i in range(1, cu_cur.shape[0]):
            for j in range(0, cu_cur.shape[1]):
                if j == 0:
                    res[i, j] = cu_cur[i, j] - int((cu_cur[i-1, j] + line_left[i]) / 2)
                else:
                    res[i, j] = cu_cur[i, j] - int((cu_cur[i-1, j] + cu_cur[i, j-1]) / 2)
        return res
                                
    else:
        for i in range(0, cu_cur.shape[0]):
            for j in range(0, cu_cur.shape[1]):
                if (i == 0) and (j == 0):
                    res[i, j] = cu_cur[i, j] - int((line_left[i] + line_up[j]) / 2)
                elif (i == 0) and (j != 0):
                    res[i, j] = cu_cur[i, j] - int((cu_cur[i, j-1] + line_up[j]) / 2)
                elif (i != 0) and (j == 0):
                    res[i, j] = cu_cur[i, j] - int((cu_cur[i-1, j] + line_left[i]) / 2)
                else:
                    res[i, j] = cu_cur[i, j] - int((cu_cur[i-1, j] + cu_cur[i, j-1]) / 2)
        return res

def rgb_ver_hor_dec(res, line_up, line_left):
    '''Vertical & Horizon prediction'''
    cu_cur = np.zeros(res.shape)
    
    if (line_left[0] == -1) and (line_up[0] == -1):
        cu_cur[0, :] = diff_dec(res[0, :])
        cu_cur[:, 0] = diff_dec(res[:, 0])
        for i in range(1, res.shape[0]):
            for j in range(1, res.shape[1]):
                cu_cur[i, j] = res[i, j] + int((cu_cur[i-1, j] + cu_cur[i, j-1]) / 2)
        return cu_cur
    
    elif (line_left[0] == -1) and (line_up[0] != -1):
        cu_cur[:, 0] = diff_dec(res[:, 0])
        for i in range(0, res.shape[0]):
            for j in range(1, res.shape[1]):
                if i == 0:
                    cu_cur[i, j] = res[i, j] + int((cu_cur[i, j-1] + line_up[j]) / 2)
                else:
                    cu_cur[i, j] = res[i, j] + int((cu_cur[i-1, j] + cu_cur[i, j-1]) / 2)
        return cu_cur
    
    elif (line_left[0] != -1) and (line_up[0] == -1):
        cu_cur[0, :] = diff_dec(res[0, :])
        for i in range(1, res.shape[0]):
            for j in range(0, res.shape[1]):
                if j == 0:
                    cu_cur[i, j] = res[i, j] + int((cu_cur[i-1, j] + line_left[i]) / 2)
                else:
                    cu_cur[i, j] = res[i, j] + int((cu_cur[i-1, j] + cu_cur[i, j-1]) / 2)
        return cu_cur
                                
    else:
        for i in range(0, res.shape[0]):
            for j in range(0, res.shape[1]):
                if (i == 0) and (j == 0):
                    cu_cur[i, j] = res[i, j] + int((line_left[i] + line_up[j]) / 2)
                elif (i == 0) and (j != 0):
                    cu_cur[i, j] = res[i, j] + int((cu_cur[i, j-1] + line_up[j]) / 2)
                elif (i != 0) and (j == 0):
                    cu_cur[i, j] = res[i, j] + int((cu_cur[i-1, j] + line_left[i]) / 2)
                else:
                    cu_cur[i, j] = res[i, j] + int((cu_cur[i-1, j] + cu_cur[i, j-1]) / 2)
        return cu_cur

# G_Mean
def g_mean_enc(cu_cur, line_up, line_left):
    '''Mean prediction'''
    if (line_up[0] == -1) and (line_left[0] != -1):
        mean = np.round(np.mean(line_left))
        res = cu_cur - mean
        return res
    elif (line_left[0] == -1) and (line_up[0] != -1):
        mean = np.round(np.mean(line_up))
        res = cu_cur - mean
        return res
    elif (line_left[0] == -1) and (line_up[0] == -1):
        mean = np.round((np.power(2, 12) - 1)/2)
        res = cu_cur - mean
        return res 
    else:
        mean = np.round(np.mean(np.concatenate((line_up[:line_up.shape[0]], line_left[:line_left.shape[0]]), axis=0)))
        res = cu_cur - mean
        return res

def g_mean_dec(res, line_up, line_left):
    '''Mean prediction'''
    if (line_up[0] == -1) and (line_left[0] != -1):
        mean = np.round(np.mean(line_left))
        cu_cur = res + mean
        return cu_cur
    elif (line_left[0] == -1) and (line_up[0] != -1):
        mean = np.round(np.mean(line_up))
        cu_cur = res + mean
        return cu_cur
    elif (line_left[0] == -1) and (line_up[0] == -1):
        mean = np.round((np.power(2, 12) - 1)/2)
        cu_cur = res + mean
        return cu_cur 
    else:
        mean = np.round(np.mean(np.concatenate((line_up[:line_up.shape[0]], line_left[:line_left.shape[0]]), axis=0)))
        cu_cur = res + mean
        return cu_cur    

# R_Mean
def r_mean_enc(cu_cur, cu_G):
    res = np.zeros(cu_cur.shape)
    for i in range(0, cu_cur.shape[0]):
        for j in range(0, cu_cur.shape[1]):
            res[i, j] = cu_cur[i, j] - np.round((cu_G[i, 2*j] + cu_G[i, 2*j+1]) / 2)
    return res

def r_mean_dec(res, cu_G):
    cu_cur = np.zeros(res.shape)
    for i in range(0, res.shape[0]):
        for j in range(0, res.shape[1]):
            cu_cur[i, j] = res[i, j] + np.round((cu_G[i, 2*j] + cu_G[i, 2*j+1]) / 2)
    return cu_cur

# B_Mean
def b_mean_enc(cu_cur, cu_G, cu_R):
    res = np.zeros(cu_cur.shape)
    for i in range(0, cu_cur.shape[0]):
        for j in range(0, cu_cur.shape[1]):
            res[i, j] = cu_cur[i, j] - np.round((cu_G[i, 2*j] + cu_G[i, 2*j+1] + cu_R[i, j]) / 3)
    return res

def b_mean_dec(res, cu_G, cu_R):
    cu_cur = np.zeros(res.shape)
    for i in range(0, res.shape[0]):
        for j in range(0, res.shape[1]):
            cu_cur[i, j] = res[i, j] + np.round((cu_G[i, 2*j] + cu_G[i, 2*j+1] + cu_R[i, j]) / 3)
    return cu_cur
# ----------------------------------------------------------------------------------------------------------------------


# compute performance --------------------------------------------------------------------------------------------------
def compute_sad(res):
    res_abs = np.abs(res)
    sad = np.sum(res_abs)/(res.shape[0] * res.shape[1])
    return sad


def compute_psnr(data_orig, data_rec, bit):
    '''计算psnr的值'''
    mse = np.mean((data_orig - data_rec) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10((np.power(2, bit)-1) / np.sqrt(mse))
    return psnr


def compute_psnr_fp(data_orig, data_rec):
    '''计算psnr的值'''
    mse = np.mean((data_orig - data_rec) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(data_orig.max().max() / np.sqrt(mse))
    return psnr


def compute_bpp_all(data_orig_dir, data_enc_dir):
    for filename in os.listdir(data_orig_dir):
        file_path_orig = os.path.join(data_orig_dir, filename)
        file_path_comp = data_enc_dir + os.path.splitext(filename)[0] + '.bin'
        
        data_orig = read_csv(file_path_orig)
        num_pixels = data_orig.shape[0] * data_orig.shape[1]
        
        size_comp = os.path.getsize(file_path_comp) * 8
        bpp = size_comp / num_pixels
        print(bpp)
# ----------------------------------------------------------------------------------------------------------------------


# Encoder --------------------------------------------------------------------------------------------------------------
def encoder(R_dir, G_dir, B_dir, cu_w, cu_h, entropy_step, Q_step, R_out_dir, G_out_dir, B_out_dir):
    
    # read data from file -----------------------------------------------------#
    R_orig = read_csv(R_dir)
    G_orig = read_csv(G_dir)
    B_orig = read_csv(B_dir)
    # -------------------------------------------------------------------------#
    
    # 12bit quantization ------------------------------------------------------#
    R_norm, R_min, R_max = trans_bit_linear(R_orig, 12)
    G_norm, G_min, G_max = trans_bit_linear(G_orig, 12)
    B_norm, B_min, B_max = trans_bit_linear(B_orig, 12)
    # -------------------------------------------------------------------------#
    
    # Encode G ----------------------------------------------------------------#
    encoder = deepCABAC.Encoder()

    image_size = np.concatenate((np.array([G_norm.shape[0]]), np.array([G_norm.shape[1]])), axis=0)
    encoder.encodeWeightsRD(image_size.astype(np.float32), 1.0, 144.0, 0.0)

    step = 0
    res_step = np.zeros([entropy_step, cu_w * cu_h])
    mod_step = np.zeros([entropy_step])
    
    for i in range(0, int(G_norm.shape[0]/cu_h)):
        for j in range(0, int(G_norm.shape[1]/cu_w)):
            cu_cur = G_norm[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w]
            if i == 0:
                line_up = np.full(cu_w + 1, -1)
            else:
                line_up = G_norm[i*cu_h-1, j*cu_w:(j+1)*cu_w]
                    
            if j == 0:
                line_left = np.full(cu_h + 1, -1)
            else:
                line_left = G_norm[i*cu_h:(i+1)*cu_h, j*cu_w-1]
                
            res_1 = rgb_ver_enc(cu_cur, line_up)
            res_2 = rgb_hor_enc(cu_cur, line_left)
            res_3 = rgb_ver_hor_enc(cu_cur, line_up, line_left)
            res_4 = g_mean_enc(cu_cur, line_up, line_left)
                
            sad_1 = compute_sad(res_1)
            sad_2 = compute_sad(res_2)
            sad_3 = compute_sad(res_3)
            sad_4 = compute_sad(res_4)
            
            if (sad_1<=sad_2) & (sad_1<=sad_3) & (sad_1<=sad_4):
                for m in range(0, int(cu_h)):
                    res_step[step][m*cu_w: (m+1)*cu_w] = res_1[m]
                mod_step[step] = 0
                
            elif (sad_2<=sad_1) & (sad_2<=sad_3) & (sad_2<=sad_4):
                for m in range(0, int(cu_h)):
                    res_step[step][m*cu_w: (m+1)*cu_w] = res_2[m]
                mod_step[step] = 1
                
            elif (sad_3<=sad_1) & (sad_3<=sad_2) & (sad_3<=sad_4):
                for m in range(0, int(cu_h)):
                    res_step[step][m*cu_w: (m+1)*cu_w] = res_3[m]
                mod_step[step] = 2
                
            elif (sad_4<=sad_1) & (sad_4<=sad_2) & (sad_4<=sad_3):
                for m in range(0, int(cu_h)):
                    res_step[step][m*cu_w: (m+1)*cu_w] = res_4[m]
                mod_step[step] = 3
                        
            step = step + 1
            
            if step == entropy_step:
                encoder.encodeWeightsRD(res_step.astype(np.float32), 1.0, Q_step, 0.0) 
                encoder.encodeWeightsRD(mod_step.astype(np.float32), 1.0, 1.0, 0.0)
                res_step = np.zeros([entropy_step, cu_w * cu_h])
                mod_step = np.zeros([entropy_step]) 
                step = 0
          
    stream = encoder.finish()
    stream_bytes = stream.tobytes()
    with open(G_out_dir + '.bin', 'wb') as f_out:
        f_out.write(stream_bytes)
    # -------------------------------------------------------------------------#
    
    # Encode R ----------------------------------------------------------------#
    encoder = deepCABAC.Encoder()

    image_size = np.concatenate((np.array([R_norm.shape[0]]), np.array([R_norm.shape[1]])), axis=0)
    encoder.encodeWeightsRD(image_size.astype(np.float32), 1.0, 144.0, 0.0)

    step = 0
    res_step = np.zeros([entropy_step, cu_w * cu_h])
    mod_step = np.zeros([entropy_step])
    
    for i in range(0, int(R_norm.shape[0]/cu_h)):
        for j in range(0, int(R_norm.shape[1]/cu_w)):
            cu_cur = R_norm[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w]
            if i == 0:
                line_up = np.full(cu_w + 1, -1)
            else:
                line_up = R_norm[i*cu_h-1, j*cu_w:(j+1)*cu_w]
                    
            if j == 0:
                line_left = np.full(cu_h + 1, -1)
            else:
                line_left = R_norm[i*cu_h:(i+1)*cu_h, j*cu_w-1]
            
            cu_G = G_norm[i*cu_h:(i+1)*cu_h, 2*j*cu_w:2*(j+1)*cu_w]
                
            res_1 = rgb_ver_enc(cu_cur, line_up)
            res_2 = rgb_hor_enc(cu_cur, line_left)
            res_3 = rgb_ver_hor_enc(cu_cur, line_up, line_left)
            res_4 = r_mean_enc(cu_cur, cu_G)
                
            sad_1 = compute_sad(res_1)
            sad_2 = compute_sad(res_2)
            sad_3 = compute_sad(res_3)
            sad_4 = compute_sad(res_4)
            
            if (sad_1<=sad_2) & (sad_1<=sad_3) & (sad_1<=sad_4):
                for m in range(0, int(cu_h)):
                    res_step[step][m*cu_w: (m+1)*cu_w] = res_1[m]
                mod_step[step] = 0
                
            elif (sad_2<=sad_1) & (sad_2<=sad_3) & (sad_2<=sad_4):
                for m in range(0, int(cu_h)):
                    res_step[step][m*cu_w: (m+1)*cu_w] = res_2[m]
                mod_step[step] = 1
                
            elif (sad_3<=sad_1) & (sad_3<=sad_2) & (sad_3<=sad_4):
                for m in range(0, int(cu_h)):
                    res_step[step][m*cu_w: (m+1)*cu_w] = res_3[m]
                mod_step[step] = 2
                
            elif (sad_4<=sad_1) & (sad_4<=sad_2) & (sad_4<=sad_3):
                for m in range(0, int(cu_h)):
                    res_step[step][m*cu_w: (m+1)*cu_w] = res_4[m]
                mod_step[step] = 3
                        
            step = step + 1
            
            if step == entropy_step:
                encoder.encodeWeightsRD(res_step.astype(np.float32), 1.0, Q_step, 0.0) 
                encoder.encodeWeightsRD(mod_step.astype(np.float32), 1.0, 1.0, 0.0)
                res_step = np.zeros([entropy_step, cu_w * cu_h])
                mod_step = np.zeros([entropy_step]) 
                step = 0
          
    stream = encoder.finish()
    stream_bytes = stream.tobytes()
    with open(R_out_dir + '.bin', 'wb') as f_out:
        f_out.write(stream_bytes)
    # -------------------------------------------------------------------------#
    
    # Encode B ----------------------------------------------------------------#
    encoder = deepCABAC.Encoder()

    image_size = np.concatenate((np.array([B_norm.shape[0]]), np.array([B_norm.shape[1]])), axis=0)
    encoder.encodeWeightsRD(image_size.astype(np.float32), 1.0, 144.0, 0.0)

    step = 0
    res_step = np.zeros([entropy_step, cu_w * cu_h])
    mod_step = np.zeros([entropy_step])
    
    for i in range(0, int(B_norm.shape[0]/cu_h)):
        for j in range(0, int(B_norm.shape[1]/cu_w)):
            cu_cur = B_norm[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w]
            if i == 0:
                line_up = np.full(cu_w + 1, -1)
            else:
                line_up = B_norm[i*cu_h-1, j*cu_w:(j+1)*cu_w]
                    
            if j == 0:
                line_left = np.full(cu_h + 1, -1)
            else:
                line_left = B_norm[i*cu_h:(i+1)*cu_h, j*cu_w-1]
            
            cu_G = G_norm[i*cu_h:(i+1)*cu_h, 2*j*cu_w:2*(j+1)*cu_w]
            cu_R = R_norm[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w]
                
            res_1 = rgb_ver_enc(cu_cur, line_up)
            res_2 = rgb_hor_enc(cu_cur, line_left)
            res_3 = rgb_ver_hor_enc(cu_cur, line_up, line_left)
            res_4 = b_mean_enc(cu_cur, cu_G, cu_R)
                
            sad_1 = compute_sad(res_1)
            sad_2 = compute_sad(res_2)
            sad_3 = compute_sad(res_3)
            sad_4 = compute_sad(res_4)
            
            if (sad_1<=sad_2) & (sad_1<=sad_3) & (sad_1<=sad_4):
                for m in range(0, int(cu_h)):
                    res_step[step][m*cu_w: (m+1)*cu_w] = res_1[m]
                mod_step[step] = 0
                
            elif (sad_2<=sad_1) & (sad_2<=sad_3) & (sad_2<=sad_4):
                for m in range(0, int(cu_h)):
                    res_step[step][m*cu_w: (m+1)*cu_w] = res_2[m]
                mod_step[step] = 1
                
            elif (sad_3<=sad_1) & (sad_3<=sad_2) & (sad_3<=sad_4):
                for m in range(0, int(cu_h)):
                    res_step[step][m*cu_w: (m+1)*cu_w] = res_3[m]
                mod_step[step] = 2
                
            elif (sad_4<=sad_1) & (sad_4<=sad_2) & (sad_4<=sad_3):
                for m in range(0, int(cu_h)):
                    res_step[step][m*cu_w: (m+1)*cu_w] = res_4[m]
                mod_step[step] = 3
                        
            step = step + 1
            
            if step == entropy_step:
                encoder.encodeWeightsRD(res_step.astype(np.float32), 1.0, Q_step, 0.0) 
                encoder.encodeWeightsRD(mod_step.astype(np.float32), 1.0, 1.0, 0.0)
                res_step = np.zeros([entropy_step, cu_w * cu_h])
                mod_step = np.zeros([entropy_step]) 
                step = 0
          
    stream = encoder.finish()
    stream_bytes = stream.tobytes()
    with open(B_out_dir + '.bin', 'wb') as f_out:
        f_out.write(stream_bytes)    
    # -------------------------------------------------------------------------#
    
    return R_norm, G_norm, B_norm
    
# ----------------------------------------------------------------------------------------------------------------------


# Decoder --------------------------------------------------------------------------------------------------------------
def decoder(R_bit_dir, G_bit_dir, B_bit_dir, cu_w, cu_h, entropy_step):
    
    # Decode G ----------------------------------------------------------------#
    with open(G_bit_dir + '.bin', 'rb') as fin:
        stream = np.frombuffer(fin.read(), dtype=np.uint8)
    decoder = deepCABAC.Decoder()
    decoder.getStream(stream) 
    
    data_shape = decoder.decodeWeights()
    G_rec = np.zeros([int(data_shape[0]), int(data_shape[1])]).astype(np.int16)
    
    step = 0
    
    for i in range(0, int(data_shape[0]/cu_h)):
        for j in range(0, int(data_shape[1]/cu_w)):
            if step == 0:
                res_step = decoder.decodeWeights()
                mod_step = decoder.decodeWeights()
            
            if i == 0:
                line_up = np.full(cu_w + 1, -1)
            else:
                line_up = G_rec[i*cu_h-1, j*cu_w:(j+1)*cu_w]
                
            if j == 0:
                line_left = np.full(cu_h + 1, -1)
            else:
                line_left = G_rec[i*cu_h:(i+1)*cu_h, j*cu_w-1]
            
            mod = mod_step[step]
            res = np.zeros([cu_h, cu_w])
            for m in range(0, int(res_step[step].shape[0]/cu_w)):
                res[m] = res_step[step][m*cu_w:(m+1)*cu_w]
            
            if mod == 0:
                G_rec[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w] = rgb_ver_dec(res, line_up)
            elif mod == 1:
                G_rec[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w] = rgb_hor_dec(res, line_left)
            elif mod == 2:
                G_rec[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w] = rgb_ver_hor_dec(res, line_up, line_left)
            elif mod == 3:
                G_rec[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w] = g_mean_dec(res, line_up, line_left)
            
            step = step + 1
            
            if step == entropy_step:
                step = 0
    # -------------------------------------------------------------------------#
    
    # Decode R ----------------------------------------------------------------#
    with open(R_bit_dir + '.bin', 'rb') as fin:
        stream = np.frombuffer(fin.read(), dtype=np.uint8)
    decoder = deepCABAC.Decoder()
    decoder.getStream(stream) 
    
    data_shape = decoder.decodeWeights()
    R_rec = np.zeros([int(data_shape[0]), int(data_shape[1])]).astype(np.int16)
    
    step = 0
    
    for i in range(0, int(data_shape[0]/cu_h)):
        for j in range(0, int(data_shape[1]/cu_w)):
            if step == 0:
                res_step = decoder.decodeWeights()
                mod_step = decoder.decodeWeights()
            
            if i == 0:
                line_up = np.full(cu_w + 1, -1)
            else:
                line_up = R_rec[i*cu_h-1, j*cu_w:(j+1)*cu_w]
                
            if j == 0:
                line_left = np.full(cu_h + 1, -1)
            else:
                line_left = R_rec[i*cu_h:(i+1)*cu_h, j*cu_w-1]
            
            cu_G = G_rec[i*cu_h:(i+1)*cu_h, 2*j*cu_w:2*(j+1)*cu_w]
            
            mod = mod_step[step]
            res = np.zeros([cu_h, cu_w])
            for m in range(0, int(res_step[step].shape[0]/cu_w)):
                res[m] = res_step[step][m*cu_w:(m+1)*cu_w]
            
            if mod == 0:
                R_rec[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w] = rgb_ver_dec(res, line_up)
            elif mod == 1:
                R_rec[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w] = rgb_hor_dec(res, line_left)
            elif mod == 2:
                R_rec[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w] = rgb_ver_hor_dec(res, line_up, line_left)
            elif mod == 3:
                R_rec[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w] = r_mean_dec(res, cu_G)
            
            step = step + 1
            
            if step == entropy_step:
                step = 0    
    # -------------------------------------------------------------------------#
    
    # Decode B ----------------------------------------------------------------#
    with open(B_bit_dir + '.bin', 'rb') as fin:
        stream = np.frombuffer(fin.read(), dtype=np.uint8)
    decoder = deepCABAC.Decoder()
    decoder.getStream(stream) 
    
    data_shape = decoder.decodeWeights()
    B_rec = np.zeros([int(data_shape[0]), int(data_shape[1])]).astype(np.int16)
    
    step = 0
    
    for i in range(0, int(data_shape[0]/cu_h)):
        for j in range(0, int(data_shape[1]/cu_w)):
            if step == 0:
                res_step = decoder.decodeWeights()
                mod_step = decoder.decodeWeights()
            
            if i == 0:
                line_up = np.full(cu_w + 1, -1)
            else:
                line_up = B_rec[i*cu_h-1, j*cu_w:(j+1)*cu_w]
                
            if j == 0:
                line_left = np.full(cu_h + 1, -1)
            else:
                line_left = B_rec[i*cu_h:(i+1)*cu_h, j*cu_w-1]
            
            cu_G = G_rec[i*cu_h:(i+1)*cu_h, 2*j*cu_w:2*(j+1)*cu_w]
            cu_R = R_rec[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w]
            
            mod = mod_step[step]
            res = np.zeros([cu_h, cu_w])
            for m in range(0, int(res_step[step].shape[0]/cu_w)):
                res[m] = res_step[step][m*cu_w:(m+1)*cu_w]
            
            if mod == 0:
                B_rec[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w] = rgb_ver_dec(res, line_up)
            elif mod == 1:
                B_rec[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w] = rgb_hor_dec(res, line_left)
            elif mod == 2:
                B_rec[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w] = rgb_ver_hor_dec(res, line_up, line_left)
            elif mod == 3:
                B_rec[i*cu_h:(i+1)*cu_h, j*cu_w:(j+1)*cu_w] = b_mean_dec(res, cu_G,  cu_R)
            
            step = step + 1
            
            if step == entropy_step:
                step = 0   
    # -------------------------------------------------------------------------#
    
    return R_rec, G_rec, B_rec
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    R_dir = '2024.7.21\\Panel_3_2nit_mode1_R'
    G_dir = '2024.7.21\\Panel_3_2nit_mode1_G'
    B_dir = '2024.7.21\\Panel_3_2nit_mode1_B'
    
    R_enc_dir = '2024.7.21\\Enc\\Panel_3_2nit_mode1\\R\\'
    G_enc_dir = '2024.7.21\\Enc\\Panel_3_2nit_mode1\\G\\'
    B_enc_dir = '2024.7.21\\Enc\\Panel_3_2nit_mode1\\B\\'
    
    cu_w = 8
    cu_h = 8
    entropy_step = 35640
    Q_step = 1.0
    
    for filename in os.listdir(R_dir):
        
        R_path = os.path.join(R_dir, filename)
        G_path = os.path.join(G_dir, filename)
        B_path = os.path.join(B_dir, filename)
        
        if not os.path.exists(R_enc_dir): os.makedirs(R_enc_dir)
        if not os.path.exists(G_enc_dir): os.makedirs(G_enc_dir)
        if not os.path.exists(B_enc_dir): os.makedirs(B_enc_dir)
        
        R_out_dir = os.path.join(R_enc_dir, os.path.splitext(filename)[0])
        G_out_dir = os.path.join(G_enc_dir, os.path.splitext(filename)[0])
        B_out_dir = os.path.join(B_enc_dir, os.path.splitext(filename)[0])
        
        R_norm, G_norm, B_norm = encoder(R_path, G_path, B_path, cu_w, cu_h, entropy_step, Q_step, R_out_dir, G_out_dir, B_out_dir)
        R_rec, G_rec, B_rec = decoder(R_out_dir, G_out_dir, B_out_dir, cu_w, cu_h, entropy_step)

        psnr_R = compute_psnr(R_norm, R_rec, 12)
        psnr_G = compute_psnr(G_norm, G_rec, 12)
        psnr_B = compute_psnr(B_norm, B_rec, 12)
        
        print('psnr_R of', filename, ' = ', psnr_R)
        print('psnr_G of', filename, ' = ', psnr_G)
        print('psnr_B of', filename, ' = ', psnr_B)
    
    print('\n')
    print('R240/R32/R64:')    
    compute_bpp_all(R_dir, R_enc_dir)
    
    print('\n')
    print('G240/G32/G64:')   
    compute_bpp_all(G_dir, G_enc_dir)
    
    print('\n')
    print('B240/B32/B64:')   
    compute_bpp_all(B_dir, B_enc_dir)
    
    




    

