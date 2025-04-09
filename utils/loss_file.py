import pandas as pd
from datetime import datetime
import os

def save_loss(
    t_loss=0, 
    t_SegLoss=0,
    t_PatchCLLoss=0, 
    t_miou=0,    
    t_dice=0,
    t_consistency_Loss=0,
    v_loss=0, 
    v_SegLoss=0,
    v_PatchCLLoss=0, 
    v_miou=0,    
    v_dice=0,
    filename='loss_record.csv'
):
    # 獲取文件的目錄部分
    file_dir = os.path.dirname(filename)
    
    # 如果目錄不存在，創建目錄
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        print(f"Created directory: {file_dir}")
    
    # 创建一个包含损失值和当前时间的字典
    data = {
        't_loss': [t_loss], 
        't_SegLoss': [t_SegLoss], 
        't_PatchCLLoss': [t_PatchCLLoss], 
        't_miou':[t_miou],
        't_dice':[t_dice],
        't_consistency_Loss':[t_consistency_Loss],
        'v_loss': [v_loss], 
        'v_SegLoss': [v_SegLoss], 
        'v_PatchCLLoss': [v_PatchCLLoss], 
        'v_miou':[v_miou],
        'v_dice':[v_dice],
        'time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }
    
    # 检查文件是否存在
    if not os.path.exists(filename):
        # 如果文件不存在，创建一个新的 DataFrame，并保存为 CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Created new file and saved data: {filename}")
    else:
        # 如果文件已存在，加载文件，添加新数据，并保存
        df = pd.read_csv(filename)
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(filename, index=False)
        print(f"Appended new data and saved to existing file: {filename}")


def check_loss_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"File {filename} has been deleted.")
    else:
        print(f"File {filename} does not exist in the directory.")
