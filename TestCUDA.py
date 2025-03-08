import torch

print(torch.cuda.is_available())          # 是否支持 CUDA
print(torch.cuda.device_count())          # 可用 GPU 数量
print(torch.cuda.current_device())        # 当前使用的 GPU 索引
print(torch.cuda.get_device_name(0))      # 索引 0 的 GPU 名称（核显？）
print(torch.cuda.get_device_name(1))      # 索引 1 的 GPU 名称（独显？）