from cProfile import label
from ctypes import sizeof
from sympy import fu
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 1.把rnn模型写出来，还得储存正弦函数的数据
# 2.训练模型的函数
#3.测试模型的函数
#4.main函数
#5.画图

class RNN(nn.Module):
    def __init__(self,input_size=1,hidden_size=32,output_size = 1,num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first= True
            )

        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        #需要初始化隐藏矩阵     x:batch_size *sequence_size *  input_size
        h = torch.zeros(
            (self.num_layers,x.size(0),self.hidden_size),
            device=x.device) #num_layers * batch_size * hidden_size 
        output,_ = self.rnn(x,h)  #output: batch-size * sequence_size * outputsize
        output = output[:,-1,:]     #只需要获取最后时间步数的输出
        output = self.fc(output)
        return output

def generate_sine_wave(seq_length = 100,num_samples = 1000):
    time_steps = 1000
    time = np.arange(0,time_steps,1)
    data = np.sin(0.1 * time) + np.random.normal(0,0.1,size=len(time))
    x = np.zeros((num_samples,seq_length,1))
    y = np.zeros((num_samples,1))
    #x: 记录了前面seq_length个的函数值，y: 记录了对应索引的sin函数值
    #num_samples:是需要记录的数据量，seq_length: 是预测数据需要参考的前面的数据个数，即时间步数
    for i in range(num_samples):
        start_idx = np.random.randint(0, len(time) - seq_length - 1)
        # 输入序列
        x[i, :, 0] = data[start_idx:start_idx+seq_length]
        # 预测下一个值 - 修复：应该是seq_length而不是seq_length-1
        y[i, 0] = data[start_idx + seq_length]

    x_tensor = torch.FloatTensor(x)
    y_tensor = torch.FloatTensor(y)
    
    # 打印第一个样本的最后一个输入值和目标输出值，用于验证
    print(f"输入序列末值: {x[0, -1, 0]:.4f}, 目标输出: {y[0, 0]:.4f}")
    
    return x_tensor, y_tensor


def train_rnn_model(model,x_train,y_train,lr =0.01,epochs = 100):
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    
    # 确保数据在正确的设备上
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    # 设置模型为训练模式
    model.train()
    
    for epoch in range(epochs):
        
        
        # 前向传播
        output = model(x_train)
        loss = criterion(output, y_train)
        
        # 反向传播和优化
        # 清零梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失
        losses.append(loss.item())
        
        # 打印训练进度
        if epoch % 20 == 0 or epoch == epochs-1:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.8f}")

    return losses

def test_model(model,seq_length = 50,future_steps = 50):
    device = next(model.parameters()).device
    time = np.arange(0,seq_length + future_steps,1)
    true_data = np.sin(0.1*time) + np.random.normal(0,0.05,len(time))
    input_seq = true_data[:seq_length]#x:batch_size * seq_len * input-size
    x = torch.FloatTensor(input_seq).view(1,seq_length,1).to(device)
    
    predictions = []
    
    # 设置模型为评估模式
    model.eval()
    
    with torch.no_grad():
        current_input = x
        for _ in range(future_steps):
            # 获取预测
            pred = model(current_input)
            predictions.append(pred.item())
            
            # 更新输入序列 (移除最早的点，添加预测的点)
            new_input = torch.cat([current_input[:, 1:, :], pred.view(1, 1, 1)], dim=1)
            current_input = new_input
    
    # 改进可视化
    plt.figure(figsize=(14, 7))
    
    # 绘制历史数据
    plt.plot(time[:seq_length], true_data[:seq_length], 'b-', linewidth=2, label="历史数据")
    
    # 绘制真实未来数据
    plt.plot(time[seq_length:], true_data[seq_length:], 'g-', linewidth=2, label="真实未来")
    
    # 绘制预测数据
    future_time = time[seq_length:seq_length+future_steps]
    plt.plot(future_time,predictions,'r--',label="predicted")
    
    plt.title("RNN predict SIN")
    plt.xlabel("x")
    plt.ylabel("sinx")
    plt.legend()
    plt.grid(True)
    plt.savefig('predict.png')
    plt.close()
    
    return true_data, predictions

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 超参数设置
    seq_length = 100
    num_layer = 2
    hidden_size = 64
    epochs = 140

    model = RNN(hidden_size=hidden_size,num_layers=num_layer).to(device)
    x_train,y_train = generate_sine_wave()
    print("----------start to train------")
    losses = train_rnn_model(model,x_train,y_train,epochs=epochs)
    print("\n测试模型并预测未来...")
    true_data,predictions = test_model(model,seq_length=seq_length)
