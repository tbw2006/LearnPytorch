from matplotlib.cbook import index_of
from pandas.core.internals.concat import NullArrayProxy
import torch
import torch.nn as nn
import torch.optim as optim
import re
from collections import Counter
import numpy as np


class textRNN(nn.Module):
    def __init__(self,vocab_size,embedding_dim,input_size =1,hidden_size=32,output_size=1,num_layers = 1,dropout = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers= num_layers
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)

        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers= num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size,vocab_size)

    def forward(self,x):
        output = self.embedding(x)
        
        # 检查输入是否为批量输入
        if len(x.shape) == 1:  # 非批量输入 [seq_len]
            batch_size = 1
            # 为LSTM创建两个隐藏状态(h0和c0)
            h0 = torch.zeros(
                (self.num_layers, batch_size, self.hidden_size),
                device=x.device)
            c0 = torch.zeros(
                (self.num_layers, batch_size, self.hidden_size),
                device=x.device)
            # 添加批次维度
            output = output.unsqueeze(0)  # [1, seq_len, embedding_dim]
            output, _ = self.rnn(output, (h0, c0))
            output = output[:, -1, :]  # [1, hidden_size]
            output = self.fc(output)
            return output.squeeze(0)  # 移除批次维度 [output_size]
        else:  # 批量输入 [batch_size, seq_len]
            batch_size = x.size(0)
            # 为LSTM创建两个隐藏状态(h0和c0)
            h0 = torch.zeros(
                (self.num_layers, batch_size, self.hidden_size),
                device=x.device)
            c0 = torch.zeros(
                (self.num_layers, batch_size, self.hidden_size),
                device=x.device)
            output, _ = self.rnn(output, (h0, c0))
            output = output[:, -1, :]  # [batch_size, hidden_size]
            output = self.fc(output)
            return output  # [batch_size, output_size]


#我想要生成一些随机的片段给他作为训练
def generate_train_text(words,word_to_idx,seq_len,num_samples):#words:已经分好词的text
     # 确保有足够的文本生成样本
    max_idx = len(words) - seq_len
    if max_idx <= 0:
        raise ValueError(f"文本长度({len(words)})必须大于序列长度({seq_len})")
    
    # 初始化输入和目标张量
    text_index = switch_text_index(word_to_idx,words)
    
    max_idx = len(words) - seq_len 
    train_text = np.zeros((num_samples,seq_len),dtype=np.int64)
    target_words = np.zeros((num_samples),dtype=np.int64)
    for i in range(num_samples):
        idx = np.random.randint(0,max_idx)
        train_text[i,:] = text_index[idx:idx+seq_len]
        target_words[i] = text_index[idx+seq_len]

    train_tensor = torch.LongTensor(train_text)     #使用long因为索引都是整数
    target_tensor = torch.LongTensor(target_words)
    return train_tensor,target_tensor



def train_model(model,words, seq_len, word_to_idx,device,lr = 0.001,epochs = 10):
    optimizer = optim.Adam(model.parameters(),lr = lr)
    criterion = nn.CrossEntropyLoss()
    train_tensor,target_tensor = generate_train_text(
        words,
        word_to_idx=word_to_idx,
        seq_len=seq_len,
        num_samples=num_samples)
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

def preprocess_text(text):
    #用于将文本预处理一下，不要回车，不要标点符号
    text = text.lower()         #去掉大小写
    text = text.replace('\n',' ')#把回车变成空格
    text = re.sub(r'[^\w\s.,;\-\'?":!"]',' ',text)
    #把除了字母以及必要的标点符号以外的变成空格   '前面要加\否则就会认为字符串提前结束
    text = re.sub(r'\s+',' ',text)#把多个空格变成一个
    return text

def build_vocab(text,max_vocab_size = 10000):
    words = text.split()   #分词
    word_count = Counter(words)#把每个词的频率拿出来
    common_words = word_count.most_common(max_vocab_size-2)#把最频繁的前max_vocab_size个拿出来
    idx_to_word = {0:'<PAD>',1:'<UNK>'}#初始化词典
    word_to_idx = {'<PAD>':0,'<UNK>':1}
    for i,(word,_) in enumerate(common_words):#得到词汇表
        idx = i+2
        idx_to_word[idx] = word
        word_to_idx[word] = idx

    return word_to_idx,idx_to_word



def switch_text_index(word_to_idx,words):
    text_index = [word_to_idx.get(word,word_to_idx.get('<UNK>',0))for word in words]
    return text_index


def switch_index_text(idx_to_word,output_tensor):
    idx = torch.argmax(output_tensor).item()
    output_ch = idx_to_word.get(idx,idx_to_word.get('<UNK>',0))
    return output_ch


def predict_text(model, words, seq_len, pred_len, word_to_idx, idx_to_word, device, temperature=0.7):
    model.eval()  # 设置为评估模式
    
    # 将文本转换为索引
    text_indices = switch_text_index(word_to_idx=word_to_idx, words=words)
    
    # 确保文本长度足够
    if len(text_indices) < seq_len:
        print(f"警告：文本长度({len(text_indices)})小于序列长度({seq_len})，将使用填充")
        # 使用第一个字符索引填充
        text_indices = [text_indices[0]] * (seq_len - len(text_indices)) + text_indices
    
    # 使用文本的最后seq_len个字符作为初始输入
    input_seq = text_indices[-seq_len:]
    
    # 存储生成的字符
    generated_words = []
    
    # 生成pred_len个word
    with torch.no_grad():  # 不计算梯度，提高推理速度
        for _ in range(pred_len):
            # 准备输入张量
            input_tensor = torch.LongTensor(input_seq).to(device)
            
            # 获取模型输出
            output = model(input_tensor)
            
            # 使用温度参数调整概率分布
            scaled_logits = output / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=0)
            
            # 使用概率分布进行随机采样，而不是选择最高概率
            next_idx = torch.multinomial(probs, 1).item()
            pred_word = idx_to_word.get(next_idx, '<UNK>')
            
            # 将预测的字符添加到生成列表
            generated_words.append(pred_word)
            
            # 更新输入序列：移除第一个字符，添加预测的字符
            input_seq = input_seq[1:] + [next_idx]
    
    # 将生成的字符连接成字符串
    return generated_words


def read_text_file(file_path):
   
    try:
        with open(file_path,'r',encoding='utf-8')as file:
            text = file.read()
        return text

    except UnicodeDecodeError:
        with open(file_path,'r',encoding='latin-1')as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"error when reading file:{e}")
        return
    

if  __name__ == "__main__":
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    # 使用预定义的示例文本，而不是等待用户输入
    #text = "The quick brown fox jumps over the lazy dog. Python programming is fun and educational. Learning deep learning with PyTorch is an exciting journey. Recurrent neural networks are powerful for sequential data processing."
    #print(f"使用示例文本: {text[:50]}...")
    file_path = r"I:\Data\Nlp\PrideAndPrejudice.txt"
    text = read_text_file(file_path)
    text = preprocess_text(text)
    words = text.split()
    seq_len = 50
    num_samples = 2000
    lr = 0.001
    epochs = 5
    pred_len = 300
    vocab_size = 10000  # 词汇表大小
    word_to_idx,idx_to_word = build_vocab(text,max_vocab_size=vocab_size)
    vocab_size = len(idx_to_word)
    embedding_dim = 64  # 嵌入维度
    hidden_size = 512 # 隐藏层大小
    num_layers = 3
    model = textRNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers).to(device)
    
    losses = train_model(
        model,
        words,
        seq_len=seq_len,
        word_to_idx=word_to_idx,
        device=device,
        lr=lr,
        epochs=epochs)
    output_text = predict_text(model,words,seq_len,pred_len,word_to_idx,idx_to_word,device)
    print("below are predicted text")
    # 将单词列表用空格连接成一个句子
    
    print(' '.join(output_text))