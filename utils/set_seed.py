import torch
import numpy as np
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Function to set seed
def seed_torch(seed):
    random.seed(seed)  # 設定 Python 標準隨機庫
    np.random.seed(seed)  # 設定 NumPy 隨機數生成器
    torch.manual_seed(seed)  # 設定 PyTorch CPU 隨機數生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 設定 PyTorch CUDA 隨機數生成器
        torch.cuda.manual_seed_all(seed)  # 設定所有 GPU 設備的隨機數生成器
        torch.backends.cudnn.deterministic = True  # 確保 CuDNN 使用固定的算法
        torch.backends.cudnn.benchmark = False  # 關閉 CuDNN 自適應算法選擇    