import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


train_transform = transforms.Compose([
    transforms.RandomCrop(32,padding=4),#四周填充4个像素，再随机裁剪出32*32
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))

])

test_transform = transforms.Compose([
    transforms.Pad(4),  # 填充4像素，保持与训练一致
    transforms.CenterCrop(32),  # 确定性中心裁剪
    transforms.RandomHorizontalFlip(),#随机翻转
    #transforms.ColorJitter(brightness=0.2,contrast=0.2),#改变颜色
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))  # 添加归一化
])

train_dataset = datasets.CIFAR10(root='./CIFAR10',
    train=True,
    download=True,
    transform=train_transform)

test_dataset = datasets.CIFAR10(
    root='./CIFAR10',
    train=False,
    download=True,
    transform=test_transform
    )


batch_size = 64

train_loader = DataLoader(
    train_dataset,          #数据
    batch_size= batch_size,#批次样本数量
    shuffle = True#是否打乱
    )

test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False
    )

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 定义ResNet架构
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 残差块组
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        
        # 全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

    def _make_layer(self, channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, channels, stride))
        self.in_channels = channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def evaluate(test_loader,model):
    total = 0
    right = 0
    with torch.no_grad():
        model.eval()
        for x,y in test_loader:
            outputs = model.forward(x)
            predicted = torch.argmax(outputs,dim=1)
            right += (predicted == y).sum().item()
            total += y.size(0)
    return right/total

# 其他代码保持不变...
if __name__ == '__main__':
    # 初始化模型
    model = ResNet()
    
    # 优化器配置
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

    # 训练循环
    for epoch in range(30):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # 验证和调度
        accuracy = evaluate(test_loader, model)
        scheduler.step()
        print(f"Epoch {epoch+1}: Test Acc {accuracy:.2%}")