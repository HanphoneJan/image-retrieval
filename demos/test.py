import torch
print(torch.__version__)
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"检测到的CUDA设备数量: {torch.cuda.device_count()}")
print(torch.version.cuda)