from torch import tanh
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.w_ih = nn.Parameter(torch.randn(input_size,hidden_size))   #ih：input->hidden
        self.w_hh = nn.Parameter(torch.randn(hidden_size,hidden_size))  #hh: hidden->hidden
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))      #w:权重矩阵，b:偏置
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))


    def forward(self,x,h_prev):
        #sizeof(x): batchsize*inputsize
        h_t = tanh(
            x @ self.w_ih + self.b_ih           #使用广播自动使得规模对的上
            + h_prev @ self.w_hh + self.b_hh    #输出 batchsize * hiddesize
        )
        return h_t
seq_len = 5    # 序列长度
batch_size = 2  # 批处理大小
input_size = 3  # 输入特征维度
hidden_size = 4
model = RNN(input_size=3,hidden_size=4)
inputs = torch.randn(seq_len,batch_size,input_size)
h = torch.zeros(batch_size,hidden_size)#初始的隐藏矩阵
outputs = []
for t in range(seq_len):
    h = model(inputs[t],h)
    outputs.append(h)

outputs=torch.stack(outputs)
print("手动RNN输出形状:", outputs.shape)



