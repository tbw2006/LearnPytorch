import torch
import torch.nn as nn
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1=nn.Linear(4,6)
        self.fc2=nn.Linear(6,2)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 初始化网络
model = SimpleNet()

# 随机生成输入数据（2个样本，每个样本4个特征）
inputs = torch.randn(2, 4)  # 形状：[batch_size, input_size]

# 前向传播
outputs = model(inputs)
print(outputs)
# 输出形状：[2, 2]，表示2个样本，每个样本对应2个类别的得分