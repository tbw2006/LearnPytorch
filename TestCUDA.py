import torch
print(torch.cuda.is_available())          # 应输出 True
print(torch.version.cuda)                 # 应显示 12.1（PyTorch 内置的 CUDA 版本）
print(torch.cuda.get_device_name(0))      # 应显示 "GeForce GTX 950M"