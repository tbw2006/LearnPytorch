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
    train_dataset,
    batch_size= batch_size,
    shuffle = True
    
    )

test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False
    )

class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3,60)
        self.fc2 =  nn.Linear(60,60)
        self.fc3 = nn.Linear(60,60)
        self.fc4 = nn.Linear(60,10)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

def evaluate(test_loader,model):
    total = 0
    right = 0
    with torch.no_grad():
        for x,y in test_loader:
            outputs = model.forward(x)
            for i,output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    right += 1
                total += 1
    return right/total


if __name__ == '__main__':
    model = Model() 
    optimszer = torch.optim.Adam(model.parameters(),lr = 0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(2):
        for x , y in train_loader:
            model.zero_grad()
            outputs = model.forward(x)
            loss = criterion(outputs,y)
            loss.backward()
            optimszer.step()

        print("epoch", epoch, "accuracy", evaluate(test_loader,model))
            


