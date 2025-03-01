import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

model = SimpleNet()
x = torch.randn(1000,4)
y= torch.randint(0,2,(1000,))
dataset=TensorDataset(x,y)
train_loader = DataLoader(dataset,batch_size=32,shuffle=True)
criterion = nn.CrossEntropyLoss()#损失函数
optimizer = optim.Adam(model.parameters(),lr = 0.01)
simgle_sample = x[0:1]
simgle_label = y[0:1]
for epoch in range(100):
    for i, (inputs, labels) in enumerate(train_loader):
        # 前向传播
        outputs = model(simgle_sample)
        loss = criterion(outputs, simgle_label)
        
        # 反向传播与优化
        optimizer.zero_grad()  # 清空梯度（必须！否则梯度会累积）
        loss.backward()        # 计算梯度（链式法则）
        optimizer.step()       # 更新参数（根据梯度下降）
        
        #打印训练信息
        if (i+1) % 10 == 0:
           print(f'Epoch [{epoch+1}/{10}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        #print(f"Loss:{loss.item()} time:{epoch} ")


