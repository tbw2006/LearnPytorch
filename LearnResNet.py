import torch
import torch.nn as nn



in_channels = 3
out_channels = 63
stride = 1
class basic_block(nn.Module):
    def __init__ (self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        #可调整维度不同的通道和尺寸
        self.shortcut = nn.Sequential()  #创建一个空的神经网络模块容器
        if stride != 1 or in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):
        x = self.conv1(x) 
        x = self.bn1(x)
        x = torch.relu(x)  
        
        x = self.conv2(x) 
        x = self.bn2(x)
        
        x += self.shortcut(x)#残差链接
        return torch.relu(x)
    

class custom_resnet(nn.Module):
    def __init__(self,num_classes = 102):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        #堆叠残差块
        self.layer1 = self._make_layer(64,64,num_blocks = 2, stride = 1)
        self.layer2 = self._make_layer(64,128,num_blocks = 2, stride = 2)
        self.layer3 = self._make_layer(64,256,num_blocks = 2, stride = 2)


        self.pool2 = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(256,num_classes)

    def _make_layer (self,in_channels,out_channels,num_blocks,stride):
        layer = []
        layer.append(basic_block(in_channels,out_channels,stride=stride))#这些参数是给谁的
        for _ in range(1,num_blocks):
            layer.append(basic_block(out_channels,out_channels,stride = stride))
        return nn.Sequential(*layer)
    

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool2(x)
        x = x.view(1,-1)
        x = self.fc1(x)
        return x