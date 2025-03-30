#先搞一个字符级别的，以一个字符作为单位，来预测下一个字符应该是什么
#问题，怎么使用向量来表示字符，怎么定义损失函数比较好
from pandas.core.internals.concat import NullArrayProxy
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

def generate_chars(text,add_special_tokens = False):
    if text == NullArrayProxy:
        return []

    chars = sorted(list(set(text)))
    """
    从文本中提取唯一字符并按字母顺序排序
    
    Args:
        text: 输入文本字符串
        
    Returns:
        排序后的唯一字符列表
    """
    if add_special_tokens:
        special_tokens = ['<PAD>','<UNK>','<START>','<END>',]
        chars = special_tokens + chars
    return chars
    #sorted：按照字母顺序排序，list:列表，set:去掉重复的


def generate_idx_to_char(chars):
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    return char_to_idx,idx_to_char

'''
text = input("please input text for me")
chars = generate_chars(text)
char_to_idx,idx_to_char = generate_idx_to_char(chars)
print(char_to_idx)
print(idx_to_char)

'''

class textRNN(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,input_size =1,hidden_size=32,output_size=1,num_layers = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers= num_layers
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim)

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers= num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size,num_embeddings)

    def forward(self,x):
        output = self.embedding(x)
        
        # 检查输入是否为批量输入
        if len(x.shape) == 1:  # 非批量输入 [seq_len]
            batch_size = 1
            h = torch.zeros(
                (self.num_layers, batch_size, self.hidden_size),
                device=x.device)
            # 添加批次维度
            output = output.unsqueeze(0)  # [1, seq_len, embedding_dim]
            output, _ = self.rnn(output, h)
            output = output[:, -1, :]  # [1, hidden_size]
            output = self.fc(output)
            return output.squeeze(0)  # 移除批次维度 [output_size]
        else:  # 批量输入 [batch_size, seq_len]
            batch_size = x.size(0)
            h = torch.zeros(
                (self.num_layers, batch_size, self.hidden_size),
                device=x.device)
            output, _ = self.rnn(output, h)
            output = output[:, -1, :]  # [batch_size, hidden_size]
            output = self.fc(output)
            return output  # [batch_size, output_size]


#我想要生成一些随机的片段给他作为训练
def generate_train_text(text,char_to_idx,seq_len,num_samples):
     # 确保有足够的文本生成样本
    max_idx = len(text) - seq_len
    if max_idx <= 0:
        raise ValueError(f"文本长度({len(text)})必须大于序列长度({seq_len})")
    
    # 初始化输入和目标张量
    text_index = switch_text_index(char_to_idx,text)
    max_idx = len(text) - seq_len 
    train_text = np.zeros((num_samples,seq_len),dtype=np.int64)
    target_char = np.zeros((num_samples),dtype=np.int64)
    for i in range(num_samples):
        idx = np.random.randint(0,max_idx)
        train_text[i,:] = text_index[idx:idx+seq_len]
        target_char[i] = text_index[idx+seq_len]

    train_tensor = torch.LongTensor(train_text)     #使用long因为索引都是整数
    target_tensor = torch.LongTensor(target_char)
    return train_tensor,target_tensor



def train_model(model,train_tensor,target_tensor,device,lr = 0.001,epochs = 10):
    optimizer = optim.Adam(model.parameters(),lr = lr)
    criterion = nn.CrossEntropyLoss()
    train_tensor = train_tensor.to(device)
    target_tensor = target_tensor.to(device)
    losses = []
    model.train()
    for epoch in range(epochs):
        output = model(train_tensor)
        optimizer.zero_grad()
        loss = criterion(output,target_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        print(f"epoch: {epoch} , loss: {loss.item()}")
    return losses,



def switch_text_index(char_to_idx,text):
    text_index = [char_to_idx.get(ch,char_to_idx.get('<UNK>',0))for ch in text]
    return text_index

def switch_index_text(idx_to_char,output_tensor):
    idx = torch.argmax(output_tensor).item()
    output_ch = idx_to_char.get(idx,idx_to_char.get('<UNK>',0))
    return output_ch


def predict_text(model, text, seq_len, pred_len, char_to_idx, idx_to_char, device):
    model.eval()  # 设置为评估模式
    
    # 将文本转换为索引
    text_indices = switch_text_index(char_to_idx=char_to_idx, text=text)
    
    # 确保文本长度足够
    if len(text_indices) < seq_len:
        print(f"警告：文本长度({len(text_indices)})小于序列长度({seq_len})，将使用填充")
        # 使用第一个字符索引填充
        text_indices = [text_indices[0]] * (seq_len - len(text_indices)) + text_indices
    
    # 使用文本的最后seq_len个字符作为初始输入
    input_seq = text_indices[-seq_len:]
    
    # 存储生成的字符
    generated_chars = []
    
    # 生成pred_len个字符
    with torch.no_grad():  # 不计算梯度，提高推理速度
        for _ in range(pred_len):
            # 准备输入张量
            input_tensor = torch.LongTensor(input_seq).to(device)
            
            # 获取模型输出
            output = model(input_tensor)
            
            # 获取预测的下一个字符索引
            pred_idx = torch.argmax(output).item()
            pred_char = idx_to_char.get(pred_idx, '<UNK>')
            
            # 将预测的字符添加到生成列表
            generated_chars.append(pred_char)
            
            # 更新输入序列：移除第一个字符，添加预测的字符
            input_seq = input_seq[1:] + [pred_idx]
    
    # 将生成的字符连接成字符串
    return ''.join(generated_chars)


if  __name__ == "__main__":
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    # 使用预定义的示例文本，而不是等待用户输入
    #text = "The quick brown fox jumps over the lazy dog. Python programming is fun and educational. Learning deep learning with PyTorch is an exciting journey. Recurrent neural networks are powerful for sequential data processing."
    #print(f"使用示例文本: {text[:50]}...")
    text = input("input:")
    chars = generate_chars(text, add_special_tokens=True)
    seq_len = 20
    num_samples = 50
    lr = 0.001
    epochs = 100
    pred_len = 50
    char_to_idx,idx_to_char = generate_idx_to_char(chars)
    vocab_size = len(char_to_idx)  # 词汇表大小
    embedding_dim = 32  # 嵌入维度
    hidden_size = 256 # 隐藏层大小
    num_layers = 1 
    model = textRNN(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers).to(device)
    
    train_tensor,target_tensor = generate_train_text(
        text,
        char_to_idx=char_to_idx,
        seq_len=seq_len,
        num_samples=num_samples)
    losses = train_model(
        model,
        train_tensor=train_tensor,
        target_tensor=target_tensor,
        device=device,
        lr=lr,
        epochs=epochs)
    output_text = predict_text(model,text,seq_len,pred_len,char_to_idx,idx_to_char,device)
    print("below are predicted text")
    print(output_text)