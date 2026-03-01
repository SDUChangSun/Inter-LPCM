import numpy as np
from scipy.spatial import KDTree

def generate_result(A, B, lidar_num,point_number):
    """
    处理batch数据，生成最终的numpy数组
    
    参数:
        A: batch*50*5 的numpy数组
        B: 包含n*4 numpy数组的列表
        
    返回:
        batch*5*50*4 的numpy数组
    """
    batch_size = A.shape[0]
    final_output = np.zeros((batch_size, 2*lidar_num+1, 2*point_number, 4))
    
    for batch_idx in range(batch_size):
        # 1. 获取x值并确定B中的5个numpy
        x = int(A[batch_idx, -2, 3])
        start_idx = x - lidar_num
        end_idx = x + lidar_num+1  # 因为Python切片是左闭右开
        
        # 处理边界情况
        if start_idx < 0:
            # 需要补充的numpy数量
            num_to_pad = -start_idx
            # 获取可用的numpy
            available_numpys = [B[max(0, i)] for i in range(start_idx, end_idx) if i >= 0]
            # 补充前面的numpy
            padded_numpys = [B[0]] * num_to_pad + available_numpys
            selected_numpys = padded_numpys[:5]
        else:
            selected_numpys = [B[i] for i in range(start_idx, end_idx)]
        
        # 2. 获取y值并处理每个numpy
        y = A[batch_idx, -1, 0]
        for num_idx, numpy in enumerate(selected_numpys):
            # 找到最接近y的值
            col_values = numpy[:, 0]
            closest_idx = np.argmin(np.abs(col_values - y))
            
            # 确定上下范围
            start_row = max(0, closest_idx - point_number)
            end_row = min(numpy.shape[0], closest_idx + point_number)
            
            # 计算实际能获取的行数
            available_before = closest_idx - start_row
            available_after = end_row - closest_idx - 1
            
            # 如果行数不足50
            if (available_before + available_after + 1) < 2*point_number:
                # 优先补充行数少的方向
                if available_before < point_number:
                    end_row = min(numpy.shape[0], end_row + point_number-available_before)
                if available_after < point_number-1:
                    start_row = max(0, start_row - (point_number-1-available_after))
            # 再次检查
            available_before = closest_idx - start_row
            available_after = end_row - closest_idx - 1
            total = available_before + 1 + available_after
            
            # 如果还是不足，复制当前行
            if total < 2 * point_number:
                selected_rows = numpy[start_row:end_row, :]
                missing = 2 * point_number - total
    
                # 获取当前行
                current_row = numpy[closest_idx:closest_idx + 1, :]
    
                # 计算当前行在 selected_rows 中的相对位置
                relative_idx = closest_idx - start_row
    
                # 将复制的行插入到正确的位置（围绕 relative_idx）
                selected_rows = np.insert(
                    selected_rows,
                    [relative_idx] * missing,  # 在 relative_idx 位置插入 missing 次
                    np.tile(current_row, (missing, 1)),
                    axis=0 )
            else:
                selected_rows = numpy[start_row:end_row, :]
            selected_rows = selected_rows[:2 * point_number, :]
            # 存储结果
            final_output[batch_idx, num_idx, :, :] = selected_rows
    
    return final_output
