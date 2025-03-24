import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761])
])

test_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761])
])

train_set = datasets.CIFAR100(
    root='I:\Data\CIFAR100',
    train=True,
    transform=train_transform,
    download=True)

test_set = datasets.CIFAR100(
    root='I:\Data\CIFAR100',
    train=False,
    transform=test_transform,
    download=True,
)

class basic_block(nn.Module):
    def __init__(self,inc,ouc,stride = 1):  
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=inc,
            out_channels=ouc,
            kernel_size=3,
            stride=stride,
            padding=1)
        
        self.conv2 = nn.Conv2d(
            in_channels=ouc,
            out_channels=ouc,
            kernel_size=3,
            stride=1,
            padding=1)

        self.bn1 = nn.BatchNorm2d(ouc)
        self.bn2 = nn.BatchNorm2d(ouc)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.LeakyReLU(0.1)

        self.shortcut = nn.Sequential()
        if inc != ouc or stride!=1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=inc,
                    out_channels=ouc,
                    kernel_size=1,
                    stride=stride,
                    padding=0),
                nn.BatchNorm2d(ouc)
            )
        
    def forward(self,x):
        identity = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.activation(output)
        output = self.dropout(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.dropout(output)

        output += self.shortcut(identity)
        return self.activation(output)

class custom_resnet(nn.Module):
    def __init__(self, classes = 100):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layer_(64, 64, num=3, stride=1)
        self.layer2 = self._make_layer_(64, 128, num=4, stride=2)
        self.layer3 = self._make_layer_(128, 256, num=6, stride=2)
        self.layer4 = self._make_layer_(256, 512, num=3, stride=2)

        self.pool2 = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512,classes)

    def _make_layer_(self,inc,ouc,num,stride):
        layer = []
        
        layer.append(basic_block(inc=inc,ouc=ouc,stride=stride))
        for i in range(num-1):
            layer.append(basic_block(inc=ouc,ouc=ouc))
        return nn.Sequential(*layer)
    
    def forward(self,x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = torch.relu(output)
        output = self.pool1(output)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.pool2(output)
        output = output.view(output.size(0),-1)
        output = self.fc1(output)

        return output
    
train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,  # 设置为True以打乱训练数据
    pin_memory=True)

test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False,
    pin_memory=True)  # 添加pin_memory=True以加速数据传输

def evaluate(model,test_loader):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)
            output = model.forward(x)
            predicted = torch.argmax(output,dim=1)
            correct += (predicted==y).sum().item()
            total  += y.size(0)
    return correct/total

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = custom_resnet().to(device)
    
    # 使用SGD优化器，添加momentum和权重衰减
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(200):  # 增加训练轮数
        model.train()
        for x,y in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            output = model.forward(x)
            loss = criterion(output,y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        accuracy = evaluate(model=model,test_loader=test_loader)
        print(f"epoch: {epoch}, accuracy: {accuracy:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
