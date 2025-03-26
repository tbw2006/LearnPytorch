import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

"""
简单RNN示例：预测正弦波
这个例子展示了如何使用RNN预测时间序列数据
"""

# 1. 生成正弦波数据
def generate_sine_wave(seq_length=100, num_samples=1000):
    """生成正弦波数据"""
    # 创建时间点 - 修复：增加足够长的序列
    time_steps = 1000  # 确保有足够多的时间点
    time = np.arange(0, time_steps, 1)
    # 生成正弦波 (添加一些噪声)
    data = np.sin(0.1 * time) + np.random.normal(0, 0.1, size=time_steps)
    
    # 准备训练数据
    x = np.zeros((num_samples, seq_length, 1))
    y = np.zeros((num_samples, 1))
    
    # 创建样本：使用seq_length个点预测下一个点
    for i in range(num_samples):
        start_idx = np.random.randint(0, len(time) - seq_length - 1)
        x[i, :, 0] = data[start_idx:start_idx + seq_length]
        y[i, 0] = data[start_idx + seq_length]
    
    # 转换为PyTorch张量
    x_tensor = torch.FloatTensor(x)
    y_tensor = torch.FloatTensor(y)
    print(f"输入序列末值: {x[0, -1, 0]:.4f}, 目标输出: {y[0, 0]:.4f}")
    return x_tensor, y_tensor

# 2. 定义RNN模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=1):
        """
        简单的RNN模型
        
        参数:
            input_size: 输入特征的数量
            hidden_size: 隐藏状态的大小
            output_size: 输出的大小
            num_layers: RNN层的数量
        """
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN层
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # 输入和输出张量的第一个维度是batch_size
        )
        
        # 全连接层，将RNN的输出映射到预测值
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入数据, 形状为 (batch_size, sequence_length, input_size)
        
        返回:
            输出预测, 形状为 (batch_size, output_size)
        """
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播RNN
        # out形状: (batch_size, sequence_length, hidden_size)
        # 我们只需要最后一个时间步的输出
        out, _ = self.rnn(x, h0)
        
        # 获取序列的最后一个时间步
        out = out[:, -1, :]
        
        # 通过全连接层
        out = self.fc(out)
        
        return out

# 3. 训练模型
def train_rnn_model(model, x_train, y_train, epochs=100, lr=0.01):
    """训练RNN模型"""
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录训练损失
    losses = []
    
    for epoch in range(epochs):
        # 前向传播
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失
        losses.append(loss.item())
        
        # 打印进度
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return losses

# 4. 测试模型
def test_model(model, seq_length=100, future_steps=50):
    """测试模型并预测未来值"""
    # 生成一个新的正弦波作为起始序列
    time = np.arange(0, seq_length + future_steps, 1)
    true_data = np.sin(0.1 * time) + np.random.normal(0, 0.05, size=len(time))
    
    # 使用前seq_length个点来预测未来
    input_seq = true_data[:seq_length]
    
    # 转换为模型输入格式
    x_test = torch.FloatTensor(input_seq).view(1, seq_length, 1)
    
    # 存储预测结果
    predictions = []
    
    # 使用模型预测未来future_steps个点
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        # 预测第一个未来点
        current_input = x_test
        for _ in range(future_steps):
            # 获取预测
            pred = model(current_input)
            predictions.append(pred.item())
            
            # 更新输入序列 (移除最早的点，添加预测的点)
            new_input = current_input.clone()
            new_input = new_input[:, 1:, :]  # 移除第一个时间步
            new_input = torch.cat([new_input, pred.view(1, 1, 1)], dim=1)  # 添加预测值
            current_input = new_input
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    # 绘制原始数据
    plt.plot(time[:seq_length], true_data[:seq_length], 'b-', label='历史数据')
    plt.plot(time[seq_length:], true_data[seq_length:], 'g-', label='真实未来')
    
    # 绘制预测
    future_time = time[seq_length:seq_length+future_steps]
    plt.plot(future_time, predictions, 'r--', label='RNN预测')
    
    plt.title('RNN时间序列预测')
    plt.xlabel('时间')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)
    plt.savefig('rnn_prediction.png')
    plt.close()
    
    return true_data, predictions

# 主函数
if __name__ == "__main__":
    # 参数设置
    seq_length = 50  # 序列长度
    hidden_size = 64  # 隐藏层大小
    num_layers = 2   # RNN层数
    epochs = 200     # 训练轮数
    
    # 生成数据
    print("生成训练数据...")
    x_train, y_train = generate_sine_wave(seq_length=seq_length)
    
    # 创建模型
    model = SimpleRNN(
        input_size=1, 
        hidden_size=hidden_size, 
        output_size=1, 
        num_layers=num_layers
    )
    
    # 训练模型
    print("\n开始训练模型...")
    losses = train_rnn_model(model, x_train, y_train, epochs=epochs)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('rnn_training_loss.png')
    plt.close()
    
    # 测试模型
    print("\n测试模型并预测未来...")
    true_data, predictions = test_model(model, seq_length=seq_length, future_steps=100)
    
    print("\n完成! 预测结果已保存为图片。")
    
    # 保存模型
    torch.save(model.state_dict(), 'simple_rnn_model.pth')
    print("模型已保存为 'simple_rnn_model.pth'")
