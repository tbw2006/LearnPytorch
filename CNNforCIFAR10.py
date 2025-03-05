import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
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

class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        self.fc1 = nn.Linear(64*8*8,256)
        self.fc2 = nn.Linear(256,10)
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1,64*8*8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

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


if __name__ == '__main__':
    model = Model() 
    optimszer = torch.optim.Adam(model.parameters(),lr = 0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimszer,'max',patience=3)
    for epoch in range(5):
        model.train()
        for x , y in train_loader:
            optimszer.zero_grad()
            outputs = model.forward(x)
            loss = criterion(outputs,y)
            loss.backward()
            optimszer.step()
        accuracy = evaluate(test_loader,model)
        scheduler.step(accuracy)
        print("epoch", epoch, "accuracy", accuracy)
            


