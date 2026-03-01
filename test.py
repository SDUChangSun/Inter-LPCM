import torch
from torch import optim, nn
from model import MyModel
from r_model import PointCloudTransformer
import os
import numpy as np 
from data_processing import readbin, save_list_to_txt
import glob,datetime
import pt as pointCloud
import random
from dataprepare.data import dataPrepare, spherical_to_cartesian, psnr
#from rlconding import run_length_encode
import deepCABAC
from nearpoint import generate_result
from r_entropy_model import r_entropymodel
import math
import torchac
from phi_entropy_model import phi_entropymodel

def normalize_data(data, x):
    """
    对输入数据进行归一化，假设数据形状为 (N, 4)。
    Args:
        data: 输入数据，形状 (N, 4)，4个维度的范围分别为:
            - 第0维: [0, 360/x]
            - 第1维: [-5, 20]
            - 第2维: [0, 80]
            - 第3维: [0, 63]
        x: 第一维的分母参数，默认为2（即范围是[0, 180]）。
    Returns:
        归一化后的数据，范围[0, 1]。
    """
    normalized_data = torch.zeros_like(data, dtype=torch.float32)
    
    # 第0维: [0, 360/x] -> [0, 1]
    normalized_data[:,:, 0] = data[:,:, 0]*x/4
    
    # 第1维: [-5, 20] -> [0, 1]
    normalized_data[:,:, 1] = (data[:,:, 1] + 30)
    
    # 第2维: [0, 80] -> [0, 1]
    normalized_data[:,:, 2] = data[:,:, 2]
    if data.shape[2] ==4:
    # 第3维: [0, 63] -> [0, 1]
        normalized_data[:,:, 3] = data[:,:, 3]
    
    return normalized_data

def normalize_data2(data, x):
    """
    对输入数据进行归一化，假设数据形状为 (N, 4)。
    Args:
        data: 输入数据，形状 (N, 4)，4个维度的范围分别为:
            - 第0维: [0, 360/x]
            - 第1维: [-5, 20]
            - 第2维: [0, 80]
            - 第3维: [0, 63]
        x: 第一维的分母参数，默认为2（即范围是[0, 180]）。
    Returns:
        归一化后的数据，范围[0, 1]。
    """
    normalized_data = torch.zeros_like(data, dtype=torch.float32)
    
    # 第0维: [0, 360/x] -> [0, 1]
    normalized_data[:,:,:, 0] = data[:,:,:, 0]*x/4
    
    # 第1维: [-5, 20] -> [0, 1]
    normalized_data[:,:,:, 1] = (data[:,:,:, 1] + 30)
    
    # 第2维: [0, 80] -> [0, 1]
    normalized_data[:,:,:, 2] = data[:,:,:, 2]
    if data.shape[3] ==4:
    # 第3维: [0, 63] -> [0, 1]
        normalized_data[:,:,:, 3] = data[:,:,:, 3]
    
    return normalized_data     


def get_train_data(train_data,train_data2,ref_train_data,phi_step):
    train_data = train_data.to(device)
    train_data = torch.squeeze(train_data,0)
    total_label = train_data[:,-1, 4]
    train_data2 = train_data2.to(device)
    train_data2 = torch.squeeze(train_data2,0)
    ref_train_data = ref_train_data.to(device)
    ref_train_data = torch.squeeze(ref_train_data,0)
    train_data = normalize_data(train_data, phi_step)
    train_data2 = normalize_data(train_data2, phi_step)
    ref_train_data = normalize_data2(ref_train_data, phi_step)
    return train_data,train_data2,ref_train_data,total_label
    
def get_train_data2(input_tensor):
    reshaped = input_tensor.view(input_tensor.shape[0], 7, 10, 4)
    diff = reshaped[:, :, 1:, 2] - reshaped[:, :, :-1, 2]  # shape: [n, 7, 9]
    output = reshaped[:, :, :-1, :].clone()  # 取前9行，保持原通道
    output[:, :, :, 2] = diff  # 将差分结果填入第2通道
    final_output = output.view(input_tensor.shape[0], 7 * 9, 4)  # 或直接 reshape(n, 63, 4)
    return final_output


def encoder_phi(p,a_e,j):

    bit_fangwei = np.array([])
    for lidar in p:
        lidar = lidar
        res_fangwei = lidar[1:,0]-lidar[:-1,0]
        bit_fangwei = np.append(bit_fangwei,lidar[0,0])
        bit_fangwei = np.append(bit_fangwei,res_fangwei)
                #bit_fangwei = np.append(bit_fangwei,-100)
    bit_fangwei_int = bit_fangwei.astype(np.int16)
    #rl_coding_data = run_length_encode(bit_fangwei_int, fangwei_step)
    encoder = deepCABAC.Encoder()
    encoder.encodeWeightsRD(bit_fangwei[0:65535].astype(np.float32), 1.0, Q_step, 0.0)
    encoder.encodeWeightsRD(bit_fangwei[65535:].astype(np.float32), 1.0, Q_step, 0.0)
    stream = encoder.finish()
    stream_bytes = stream.tobytes()
    if a_e:
        with open('step/'+str(j) + '.bin', 'wb') as f_out:
            filename4 = 'step/fangwei_int'+str(j)+'.txt'
            f_out.write(stream_bytes)
            save_list_to_txt(bit_fangwei_int, filename4)     
    bpip_step = len(stream_bytes)*8/bit_fangwei.shape[0]
    
    return bpip_step
def pmf_to_cdf( pmf):
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    cdf_with_0 = cdf_with_0.clamp(max=1.)

    return cdf_with_0

def likelihood2(symbols, mu, sigma, alpha):
    def skew_normal_cdf(x, mu, sigma, alpha):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if mu.dim() == 1:
            mu = mu.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
            alpha = alpha.unsqueeze(1)

        z = (x - mu) / sigma
        denom = torch.sqrt(1 + (math.pi / 8.0) * alpha * alpha)
        return 0.5 * (1 + torch.erf((z / denom) / math.sqrt(2)))

    lower = skew_normal_cdf(symbols - 0.5, mu, sigma, alpha)
    upper = skew_normal_cdf(symbols + 0.5, mu, sigma, alpha)
    pmf = (upper - lower).clamp(min=1e-9)
    return pmf


def likelihood(symbols,mu,sigma):
    #print(symbols.shape)
    #print(mu.shape)
    def logistic_cdf(x, mu, sigma):
    # 确保x和mu/sigma的形状兼容
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1578] -> [1, 1578]
        if mu.dim() == 1:
            mu = mu.unsqueeze(1)  # [126778] -> [126778, 1]
            sigma = sigma.unsqueeze(1)
    
        #return torch.sigmoid((x - mu) / sigma)
        return 0.5 * (1 + torch.erf((x - mu) / (sigma * math.sqrt(2))))
        # compute pmf as CDF difference between bins ±0.5
    lower = logistic_cdf(symbols - 0.5, mu,sigma)
    upper = logistic_cdf(symbols + 0.5, mu,sigma)
    pmf = (upper - lower).clamp(min=1e-9)  # [N, C, K]
    return pmf
    
def encoder_phi3(p, a_e, j, model, phi_step, coding_length):
    likelihood_bound = 1e-9
    bit_fangwei_list = []
    lengths = np.array([len(arr) - 1 for arr in p])

    # 1. 构建 bit_fangwei：角度差序列
    for lidar in p:
        res_fangwei = lidar[1:, 0] - lidar[:-1, 0]
        bit_fangwei_list.append(res_fangwei)
    bit_fangwei = np.concatenate(bit_fangwei_list).astype(np.int16)
    #bit_fangwei = bit_fangwei[2200:2400]
    # 参数初始化
    seq = 50
    chunk_size = coding_length
    num_chunks = (len(bit_fangwei) + chunk_size - 1) // chunk_size
    strings_list = []
    bits = 0

    # 获取符号范围
    min_v_val = bit_fangwei.min()
    max_v_val = bit_fangwei.max()
    symbols = torch.arange(min_v_val, max_v_val + 1).reshape(1, -1).to(device)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, len(bit_fangwei))
        segment = bit_fangwei[start:end]

        # 2. 构建 padding
        if chunk_idx == 0:
            pad_value = int(np.round(0.18 / phi_step))
            padding = np.ones(seq - 1, dtype=np.int16) * pad_value
        else:
            padding = bit_fangwei[start - (seq - 1):start]
            if len(padding) < (seq - 1):
                pad_len = (seq - 1) - len(padding)
                pad_value = int(np.round(0.18 / phi_step))
                padding = np.concatenate((np.ones(pad_len, dtype=np.int16) * pad_value, padding))

        # 3. 构建 padded_segment
        padded_segment = np.append(padding, segment)

        # 4. 转换为 Tensor
        data = torch.from_numpy(padded_segment).unsqueeze(0).to(device)
        data_seq = data.squeeze(0)
        patches = data_seq.unfold(0, seq, 1)
        x_seq = patches[:, :-1].float().to(device)
        x_last = patches[:, -1].float().to(device)

        # 5. 前向                                                                                                                                                                                                                                                               1`模型推理
        model.eval()
        with torch.no_grad():
            outputs, mu, sigma,x1,alpha = model(x_seq, x_last)
        #print(mu[183:188])
        #print(x1[183:188])
        mu = torch.round(mu*100)/100
        sigma = torch.round(sigma*100)/100  
        alpha = torch.round(alpha*100)/100      
        # 6. 构建 PMF → CDF
        pmf = likelihood2(symbols.to(mu.device), mu, sigma, alpha)
        #print(pmf.shape)

        #print(sigma[183:188])
        cdf_list = pmf_to_cdf(pmf).unsqueeze(1)

        # 7. 准备编码数据
        c_data = torch.from_numpy(segment - min_v_val).to(torch.int16).unsqueeze(1)

        # 8. 熵编码
        strings = torchac.encode_float_cdf(cdf_list.cpu(), c_data, check_input_bounds=True)
        strings_list.append(strings)
        bits += len(strings)

    # 9. 计算总 bit-per-input
    bpip = bits * 8 / len(bit_fangwei)
    normalized_seq = torch.from_numpy(bit_fangwei - min_v_val).to(torch.int16).unsqueeze(1)

    return bpip, strings_list, normalized_seq, symbols, lengths, torch.tensor(min_v_val, dtype=torch.float32)

def encode_r(r_res, add_num, qp_r, three_d_array,model,coding_length):

    refine_value2 = np.empty((1,0))
    encode_res2 = np.empty((1,0))
    recont_r = np.empty((0,1))
    for i in range(three_d_array.shape[0]):
        if add_num[i]!=0:
            encode_res2 = np.hstack((encode_res2,r_res[:-add_num[i],i].reshape(1,-1)))
            recont_r = np.vstack((recont_r,three_d_array[i,49:-add_num[i],2].reshape(-1,1)))
        else:
            encode_res2 = np.hstack((encode_res2,r_res[:,i].reshape(1,-1)))
            recont_r = np.vstack((recont_r,three_d_array[i,49:,2].reshape(-1,1)))
    encode_res2 = encode_res2.reshape(-1)
    bit_fangwei = encode_res2.astype(np.int16)
    likelihood_bound = 1e-9
    bit_fangwei_list = []

    # 1. 构建 bit_fangwei：角度差序列
    # 参数初始化
    seq = 50
    chunk_size = coding_length
    num_chunks = (len(bit_fangwei) + chunk_size - 1) // chunk_size
    strings_list = []
    bits = 0

    # 获取符号范围
    min_v_val = bit_fangwei.min()
    max_v_val = bit_fangwei.max()
    symbols = torch.arange(min_v_val, max_v_val + 1).reshape(1, -1).to(device)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, len(bit_fangwei))
        segment = bit_fangwei[start:end]

        # 2. 构建 padding
        if chunk_idx == 0:
            pad_value = int(np.round(0.18 / phi_step))
            padding = np.ones(seq - 1, dtype=np.int16) * 0
        else:
            padding = bit_fangwei[start - (seq - 1):start]
            if len(padding) < (seq - 1):
                pad_len = (seq - 1) - len(padding)
                pad_value = int(np.round(0.18 / phi_step))
                padding = np.concatenate((np.ones(pad_len, dtype=np.int16) * 0, padding))

        # 3. 构建 padded_segment
        padded_segment = np.append(padding, segment)

        # 4. 转换为 Tensor
        data = torch.from_numpy(padded_segment).unsqueeze(0).to(device)
        data_seq = data.squeeze(0)
        patches = data_seq.unfold(0, seq, 1)
        x_seq = patches[:, :-1].float().to(device)
        x_last = patches[:, -1].float().to(device)

        # 5. 前向                                                                                                                                                                                                                                                               1`模型推理
        model.eval()
        with torch.no_grad():
            outputs, mu, sigma,x1 = model(x_seq, x_last)

        mu = torch.round(mu*100)/100
        sigma = torch.round(sigma*100)/100   

        # 6. 构建 PMF → CDF
        pmf = likelihood(symbols.to(mu.device), mu, sigma)
        #print(pmf.shape)

        #print(sigma[183:188])
        cdf_list = pmf_to_cdf(pmf).unsqueeze(1)

        # 7. 准备编码数据
        c_data = torch.from_numpy(segment - min_v_val).to(torch.int16).unsqueeze(1)

        # 8. 熵编码
        strings = torchac.encode_float_cdf(cdf_list.cpu(), c_data, check_input_bounds=True)
        strings_list.append(strings)
        bits += len(strings)

    # 9. 计算总 bit-per-input
    bpip = bits * 8 / len(bit_fangwei)
    normalized_seq = torch.from_numpy(bit_fangwei - min_v_val).to(torch.int16).unsqueeze(1)
    return bpip, strings_list, normalized_seq, symbols, torch.tensor(min_v_val, dtype=torch.float32),recont_r

def decoder_r(model, phi_step,strings_list,symbols,length,min_v,coding_length):
    total_lenth = length
    print(total_lenth)
    likelihood_bound = 1e-9
    padding = np.ones(49)*0
    strings = strings_list[0]
    data = torch.from_numpy(padding).unsqueeze(0).float().to(device)
    data_input = data.repeat(200,1)
    model.eval()
    with torch.no_grad():
        outputs, mu, sigma,x1 = model(data_input, data[:,-1])
    mu = torch.round(mu*100)/100

    sigma = torch.round(sigma*100)/100
    pmf = likelihood(symbols.to(mu.device), mu[0], sigma[0])  # [N, num_symbols]

    cdf_list = pmf_to_cdf(pmf)
    cdf_list = cdf_list.unsqueeze(1)

    cdf_lists = cdf_list.repeat(coding_length, 1, 1).clone()
    de_strings = torchac.decode_float_cdf(cdf_lists[0,:,:].cpu(), strings)
    print(de_strings[-1].view(1, 1).to(device)+min_v.to(device))
    data = torch.cat([data, de_strings[-1].view(1, 1).to(device)+min_v.to(device)], dim=1)
    for i in range(coding_length-1):
        data_input = data[:,-49:].repeat(200,1)
        with torch.no_grad():
            outputs, mu, sigma,x1 = model(data_input, data[:,-1])
        mu = torch.round(mu*100)/100

        sigma = torch.round(sigma*100)/100
        pmf = likelihood(symbols.to(mu.device), mu[0], sigma[0])  # [N, num_symbols]

        cdf_list = pmf_to_cdf(pmf)
        cdf_list = cdf_list.unsqueeze(1)

        cdf_lists[i+1,:,:] = cdf_list.clone() 
        de_strings = torchac.decode_float_cdf(cdf_lists[:i+2,:,:].cpu(), strings)
        print(de_strings[-1].view(1, 1).to(device)+min_v.to(device))
        data = torch.cat([data, de_strings[-1].view(1, 1).to(device)+min_v.to(device)], dim=1)
    for k in range(len(strings_list)-1):
        #print(k)
        strings = strings_list[k+1]
        data_input = data[:,-49:].repeat(200,1)
        with torch.no_grad():
            outputs, mu, sigma,x1 = model(data_input, data[:,-1])
        mu = torch.round(mu*100)/100
        sigma = torch.round(sigma*100)/100
        #if k==10:
            #print(mu)
        pmf = likelihood(symbols.to(mu.device), mu[0], sigma[0])  # [N, num_symbols]
        #pmf = torch.round(pmf*100)/100
        cdf_list = pmf_to_cdf(pmf)
        cdf_list = cdf_list.unsqueeze(1)
        #cdf_list = torch.round(cdf_list*100)/100
        cdf_lists = cdf_list.repeat(coding_length, 1, 1).clone()
        de_strings = torchac.decode_float_cdf(cdf_lists[0,:,:].cpu(), strings)
        data = torch.cat([data, de_strings[-1].view(1, 1).to(device)+min_v.to(device)], dim=1)
        if k==len(strings_list)-2:     
            for i in range(total_lenth % coding_length-1):       
                data_input = data[:,-49:].repeat(total_lenth % coding_length,1)
                with torch.no_grad():
                    outputs, mu, sigma,x1 = model(data_input, data[:,-1])
                mu = torch.round(mu*100)/100
                sigma = torch.round(sigma*100)/100
                pmf = likelihood(symbols.to(mu.device), mu[0], sigma[0])
                #pmf = torch.round(pmf*100)/100

                cdf_list = pmf_to_cdf(pmf)
                cdf_list = cdf_list.unsqueeze(1)
                #cdf_list = torch.round(cdf_list*100)/100
                cdf_lists[i+1,:,:] = cdf_list.clone() 
                de_strings = torchac.decode_float_cdf(cdf_lists[:i+2,:,:].cpu(), strings)
                data = torch.cat([data, de_strings[-1].view(1, 1).to(device)+min_v.to(device)], dim=1)
     
        else:
            for i in range(coding_length-1):       
                data_input = data[:,-49:].repeat(200,1)
                with torch.no_grad():
                    outputs, mu, sigma,x1 = model(data_input, data[:,-1])
                mu = torch.round(mu*100)/100
                sigma = torch.round(sigma*100)/100
                pmf = likelihood(symbols.to(mu.device), mu[0], sigma[0])

                cdf_list = pmf_to_cdf(pmf)
                cdf_list = cdf_list.unsqueeze(1)
                #cdf_list = torch.round(cdf_list*100)/100
                cdf_lists[i+1,:,:] = cdf_list.clone() 
                de_strings = torchac.decode_float_cdf(cdf_lists[:i+2,:,:].cpu(), strings)
                data = torch.cat([data, de_strings[-1].view(1, 1).to(device)+min_v.to(device)], dim=1)        
    return data[:,49:]-min_v

'''def encode_r(r_res, add_num, qp_r, three_d_array):

    refine_value2 = np.empty((1,0))
    encode_res2 = np.empty((1,0))
    recont_r = np.empty((0,1))
    for i in range(three_d_array.shape[0]):
        if add_num[i]!=0:
            #refine_value2 = np.hstack((refine_value2,refine_value[:-add_num[i],i].reshape(1,-1)))
            encode_res2 = np.hstack((encode_res2,r_res[:-add_num[i],i].reshape(1,-1)))
            recont_r = np.vstack((recont_r,three_d_array[i,49:-add_num[i],2].reshape(-1,1)))
        else:
            #refine_value2 = np.hstack((refine_value2,refine_value[:,i].reshape(1,-1)))
            encode_res2 = np.hstack((encode_res2,r_res[:,i].reshape(1,-1)))
            recont_r = np.vstack((recont_r,three_d_array[i,49:,2].reshape(-1,1)))
    #refine_value2 = refine_value2.reshape(-1,1)
    encode_res2 = encode_res2.reshape(-1,1)
    #recon_pt[:,1] = refine_value2[:,0]
    encoder = deepCABAC.Encoder()
    encoder.encodeWeightsRD(encode_res2[0:65535].astype(np.float32), 1.0, 1/qp_r, 0.0)
    encoder.encodeWeightsRD(encode_res2[65535:].astype(np.float32), 1.0, 1/qp_r, 0.0)
    stream = encoder.finish()
    stream_bytes = stream.tobytes()
    bpip_r = len(stream_bytes)*8/recont_r.shape[0]
    return bpip_r,recont_r,len(stream_bytes)*8'''
    
def encode_theta(r_res, add_num, qp_r, three_d_array):

    refine_value2 = np.empty((1,0))
    encode_res2 = np.empty((1,0))
    recont_r = np.empty((0,1))
    for i in range(three_d_array.shape[0]):
        if add_num[i]!=0:
            #refine_value2 = np.hstack((refine_value2,refine_value[:-add_num[i],i].reshape(1,-1)))
            encode_res2 = np.hstack((encode_res2,r_res[:-add_num[i],i].reshape(1,-1)))
            recont_r = np.vstack((recont_r,three_d_array[i,49:-add_num[i],1].reshape(-1,1)))
        else:
            #refine_value2 = np.hstack((refine_value2,refine_value[:,i].reshape(1,-1)))
            encode_res2 = np.hstack((encode_res2,r_res[:,i].reshape(1,-1)))
            recont_r = np.vstack((recont_r,three_d_array[i,49:,1].reshape(-1,1)))
    #refine_value2 = refine_value2.reshape(-1,1)
    encode_res2 = encode_res2.reshape(-1,1)
    #recon_pt[:,1] = refine_value2[:,0]
    encoder = deepCABAC.Encoder()
    encoder.encodeWeightsRD(encode_res2[0:65535].astype(np.float32), 1.0, 1/qp_r, 0.0)
    encoder.encodeWeightsRD(encode_res2[65535:].astype(np.float32), 1.0, 1/qp_r, 0.0)
    stream = encoder.finish()
    stream_bytes = stream.tobytes()
    bpip_r = len(stream_bytes)*8/recont_r.shape[0]
    return bpip_r,recont_r,len(stream_bytes)*8

def preprocess_features(features, phi_step, qp_r):
    features[:, :, 0] /= (360 / phi_step)
    features[:, :, 1] = (features[:, :, 1] + 30) / 30
    features[:, :, 2] /= 100
    features[:, :, 3] /= 64
    features[:, :, 4] /= (features[:, :, 4] + 30) / 30
    return features

def preprocess_features_single(features, phi_step):
    features[:, 0] /= (360 / phi_step)
    features[:, 1] = (features[:, 1] + 30) / 30
    features[:, 2] /= 100
    features[:, 3] /= 64
    return features


def get_three_d_array(p,seq, qp_theta,qp_r,phi_step,device,model):
    processed_lidars = []
    q = 0
    add_num = []
    numlidar = len(p)
    with torch.no_grad():
        for gt_lidar in p:
            mean = np.mean(gt_lidar,axis=0)
            lidar = np.zeros((gt_lidar.shape[0],5))
            lidar[:, 3] = q 
            lidar[:,0:3] = gt_lidar 
            #lidar[:,1] = mean[1]
            lidar[:,4] = lidar[:,1]
            q = q+1
            zero_rows = np.zeros((seq-1, 5))
            for o in range (seq-1):
                zero_rows[o,:] = lidar[0,:]-np.array([2,0,0,0,0])*(seq-1-o) 
            lidar = np.vstack((zero_rows, lidar))
            processed_lidars.append(lidar)
        max_len = max(arr.shape[0] for arr in processed_lidars)  
        for i in range(len(processed_lidars)):
            if processed_lidars[i].shape[0] < max_len:  
                zero_pad = np.zeros((max_len - processed_lidars[i].shape[0], processed_lidars[i].shape[1]))  
                processed_lidars[i] = np.vstack((processed_lidars[i], zero_pad))
                add_num.append(zero_pad.shape[0])    
            else:
                add_num.append(0) 
        three_d_array = np.stack(processed_lidars, axis=0)
    return three_d_array, add_num 

def predict_r(i, three_d_array, test_data_ora, p,ref_p,seq, qp_theta,qp_r,phi_step,device,model,split_number,ref_lidar_num, ref_point_num):
    
    test_data = test_data_ora[:split_number,:,:].copy()
    test_data2 = test_data[:,:-2,0:4].copy()
    test_data2[:,:,2] = test_data[:,1:-1,2]-test_data[:,:-2,2]
    ref_train_data = generate_result(test_data ,ref_p,ref_lidar_num, ref_point_num)
    test_data = torch.tensor(test_data, dtype=torch.float)
    test_data2 = torch.tensor(test_data2, dtype=torch.float)
    ref_train_data = torch.tensor(ref_train_data, dtype=torch.float)
    test_data,test_data2,ref_train_data, total_label = get_train_data(test_data,test_data2,ref_train_data,phi_step)     
     
    outputs1 = model(test_data[: , :-1 , :-1], test_data2,ref_train_data).squeeze()
    outputs2 = three_d_array[split_number:,i+seq-2,2].squeeze()
    
    outputs = np.concatenate([outputs1.cpu().numpy(),outputs2])
    
    res = three_d_array[:,i+seq-1,2]-outputs
    quantires = np.round(res*qp_r)
    deres = quantires/qp_r
    three_d_array[:,i+seq-1,2] = deres + outputs

    return quantires,three_d_array

def predict_r2(i, three_d_array, test_data_ora, p,ref_p,seq, qp_theta,qp_r,phi_step,device,model,split_number,ref_lidar_num, ref_point_num):
    
    outputs = three_d_array[:,i+seq-2,2].squeeze()   
    res = three_d_array[:,i+seq-1,2]-outputs
    quantires = np.round(res*qp_r)
    deres = quantires/qp_r
    three_d_array[:,i+seq-1,2] = deres + outputs

    return quantires,three_d_array

def predict_theta(i, three_d_array, test_data, p, ref_p, seq, qp_theta,qp_r,phi_step,device,model):

    test_data = torch.tensor(test_data, dtype=torch.float).to(device)
    test_data = preprocess_features(test_data, phi_step,qp_r)
    test_data2 = test_data[:, -1, :-1]
    test_data2[:,1] = test_data[:, -2, 1]
    add = three_d_array[:, end_index - 2, 1].copy()
    add = torch.from_numpy(add).to(device)
    _,outputs = model(test_data[: , :-1 , :-1], test_data2,add)
    res = three_d_array[:,end_index-1,1]-outputs.squeeze().cpu().numpy()
    quantires = np.round(res*qp_theta)
    deres = quantires/qp_theta
    three_d_array[:,end_index-1,1] = deres + outputs.squeeze().cpu().numpy()
    return deres,three_d_array      
    

if __name__ == '__main__':

    seq = 50
    oriDir = './test/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_lidar_num = 1
    ref_point_num = 10
    
    r_model = PointCloudTransformer(ref_lidar_num,hidden_dim=64, mlp_hidden_dim=128, out_dim=1).to(device)
    r_model.load_state_dict(torch.load('./r_model/37.pt', map_location='cuda:0'))
    
    input_size = 4
    num_layers = 3
    output_size =1
    hidden_size = 128
    batch_size = 1
    
    model = MyModel(input_size, hidden_size, num_layers, output_size,batch_size).to(device)
    model.load_state_dict(torch.load('./theta_model/epoch_28.pt', map_location='cuda:0'))
    r_entropy_model = r_entropymodel().to(device)
    r_entropy_model.load_state_dict(torch.load('./entropy_model/r.pt', map_location='cuda:0'))
    phi_entropy_model = phi_entropymodel().to(device)
    phi_entropy_model.load_state_dict(torch.load('./entropy_model/phi.pt', map_location='cuda:0'))
    phi_step = 0.045
    qp_r = 66
    qp_theta = 15
    Q_step = 1.0
    a_e = False
    coding_length =200
    os.makedirs('r_model', exist_ok=True)
    print(device)
    dataNames = []
     
    fileList = sorted(glob.glob(oriDir+'/*.bin'))
    for oriFile in fileList:  
        dataNames.append(oriFile)  
    total_samples = len(dataNames)
    j = 0
    for idx in range(total_samples):
        j = j+1
        filename2 = 'r_frame.bin'
        if dataNames[idx].endswith("000001.bin"):
            filename2 = dataNames[idx-1]
        if idx >= len(dataNames):
            raise IndexError(f"Index {idx} is out of bounds after adjustment.")
        print(dataNames[idx])
        p,ref_p, refPt, s_p ,inter_mode = dataPrepare(dataNames[idx], filename2, phi_step, saveMatDir='./Data/testPly')  # 确保dataPrepare函数已定义
        recon_pt = np.copy(s_p)
        bpp_phi, strings_list, normalized_seq, symbols, lengths, min_v_val = encoder_phi3(p, a_e, j,  phi_entropy_model, phi_step, coding_length)
        #inter_mode = False
        three_d_array, add_num = get_three_d_array(p, seq, qp_theta,qp_r,phi_step,device,r_model)
        
        numlidar = len(p)
        
        r_res = np.empty((0,numlidar))
        theta_res = np.empty((0,numlidar))
        with torch.no_grad():
            if inter_mode==True:
                for i in range(three_d_array.shape[1]-seq+1):
                    start_index = i  
                    end_index = i + seq 
                    test_data = three_d_array[:,start_index:end_index,:]
                    r_deres, three_d_array = predict_r(i, three_d_array, test_data, p, ref_p, seq, qp_theta,qp_r,phi_step,device,r_model,8,ref_lidar_num, ref_point_num)
                    r_res = np.vstack((r_res,r_deres))            
                    theta_deres, three_d_array = predict_theta(i, three_d_array, test_data, p, ref_p, seq, qp_theta,qp_r,phi_step,device,model)
                    theta_res = np.vstack((theta_res,theta_deres))
            else:
                for i in range(three_d_array.shape[1]-seq+1):                    
                    start_index = i  
                    end_index = i + seq 
                    test_data = three_d_array[:,start_index:end_index,:]
                    r_deres, three_d_array = predict_r2(i, three_d_array, test_data, p, ref_p, seq, qp_theta,qp_r,phi_step,device,r_model,8,ref_lidar_num, ref_point_num)
                    r_res = np.vstack((r_res,r_deres))            
                    theta_deres, three_d_array = predict_theta(i, three_d_array, test_data, p, ref_p, seq, qp_theta,qp_r,phi_step,device,model)
                    theta_res = np.vstack((theta_res,theta_deres))                       
        bpip_r, strings,n_r_res,symbols,min_v,recont_r = encode_r(r_res, add_num, qp_r, three_d_array,r_entropy_model,coding_length)
        recon_pt[:,2] = recont_r.squeeze()
        
        bpp_theta,recont_theta,_ = encode_theta(theta_res,add_num,qp_theta,three_d_array)
        recon_pt[:,1] = recont_theta.squeeze()
        
        bpip_sin = psnr(refPt, recon_pt,phi_step)
        total = bpip_r+bpp_phi+bpp_theta+bpip_sin
        print('bpip_r:',bpip_r)
        print('bpp_phi:',bpp_phi)
        print('bpp_theta',bpp_theta)
        print('bpp_sin',bpip_sin)
        print('total:',total)









