import torch
import torch.nn as nn
import torch.utils.data.dataloader
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


in_channels = 3
out_channels = 64
stride = 1
class basic_block(nn.Module):
    def __init__ (self,in_channels,out_channels,stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        #可调整维度不同的通道和尺寸
        self.shortcut = nn.Sequential()  #创建一个空的神经网络模块容器
        if stride != 1 or in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):
        identity = x
        x = self.conv1(x) 
        x = self.bn1(x)
        x = torch.relu(x)  
        
        x = self.conv2(x) 
        x = self.bn2(x)
        
        x += self.shortcut(identity)#残差链接
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
        self.layer3 = self._make_layer(128,256,num_blocks = 2, stride = 2)


        self.pool2 = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(256,num_classes)

    def _make_layer (self,in_channels,out_channels,num_blocks,stride):
        layer = []
        layer.append(basic_block(in_channels,out_channels,stride=stride))#这些参数是给谁的
        for _ in range(1,num_blocks):
            layer.append(basic_block(out_channels,out_channels,stride = 1))
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
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x
    


def evaluate(loader,model):
    total = 0
    correct = 0
    with torch.no_grad():
        count = 0
        for x,y in loader:
            if count >= 10:
                break
            x,y = x.to(device),y.to(device)
            outputs = model.forward(x)
            predicted = torch.argmax(outputs,dim=1)
            correct += (predicted==y).sum().item()
            total += y.size(0)
            count +=1
            
        return correct/total

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
)

train_dataset = datasets.ImageFolder(
    root= r'I:\Data\Flower102\flower_data\train',
    transform=transform
)

train_dataloader = DataLoader(train_dataset,batch_size = 32,shuffle = True)
if __name__ == '__main__':
    print("1111")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = custom_resnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimszer = torch.optim.Adam(model.parameters(),lr=0.01)

    for epoch in range(20):
        for x,y in train_dataloader:
            x,y = x.to(device),y.to(device)
            optimszer.zero_grad()
            outputs = model.forward(x)
            loss = criterion(outputs,y)
            loss.backward()
            optimszer.step()
        accuracy = evaluate(train_dataloader,model)
        print("epoch: ",epoch,"accuracy: ",accuracy)

        



