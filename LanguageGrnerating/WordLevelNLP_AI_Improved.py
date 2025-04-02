"""
Word-Level RNN Text Generation - AI改进版
==============================================
原始代码基础上由AI助手改进，主要增强了:
1. 模型架构 - 使用双向LSTM、层标准化和更好的全连接层
2. 训练过程 - 添加学习率调度器、早停机制和梯度裁剪
3. 文本生成 - 实现top-k、nucleus采样等高级采样策略
4. 参数优化 - 扩大嵌入维度和隐藏层大小
"""

import torch
import torch.nn as nn
import torch.optim as optim
import re
from collections import Counter
import random


class textRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, input_size=1, hidden_size=512, output_size=1, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 增加嵌入维度
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)
            
        # 使用双向LSTM增强捕捉前后文能力
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # 使用双向LSTM
        )
        
        # 添加层标准化
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # 添加全连接层，隐藏大小需要乘2因为使用了双向LSTM
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        output = self.embedding(x)  # [batch_size, seq_len, embedding_dim] 或 [seq_len, embedding_dim]
        
        # 检查输入是否为批量输入
        if len(x.shape) == 1:  # 非批量输入 [seq_len]
            batch_size = 1
            # 为双向LSTM创建两个隐藏状态(h0和c0)，注意方向数翻倍
            h0 = torch.zeros(
                (self.num_layers * 2, batch_size, self.hidden_size),
                device=x.device)
            c0 = torch.zeros(
                (self.num_layers * 2, batch_size, self.hidden_size),
                device=x.device)
            # 添加批次维度
            output = output.unsqueeze(0)  # [1, seq_len, embedding_dim]
            output, _ = self.rnn(output, (h0, c0))  # [1, seq_len, hidden_size*2]
            
            # 取最后一个时间步的输出
            last_output = output[:, -1, :]  # [1, hidden_size*2]
            
            # 应用层标准化
            last_output = self.layer_norm(last_output)  # [1, hidden_size*2]
            
            # 全连接层处理
            fc_output = torch.relu(self.fc1(last_output))  # [1, hidden_size]
            fc_output = self.dropout(fc_output)
            output = self.fc2(fc_output)  # [1, vocab_size]
            
            return output.squeeze(0)  # 移除批次维度 [vocab_size]
        else:  # 批量输入 [batch_size, seq_len]
            batch_size = x.size(0)
            # 为双向LSTM创建两个隐藏状态(h0和c0)，注意方向数翻倍
            h0 = torch.zeros(
                (self.num_layers * 2, batch_size, self.hidden_size),
                device=x.device)
            c0 = torch.zeros(
                (self.num_layers * 2, batch_size, self.hidden_size),
                device=x.device)
            output, _ = self.rnn(output, (h0, c0))  # [batch_size, seq_len, hidden_size*2]
            
            # 取最后一个时间步的输出
            last_output = output[:, -1, :]  # [batch_size, hidden_size*2]
            
            # 应用层标准化
            last_output = self.layer_norm(last_output)  # [batch_size, hidden_size*2]
            
            # 全连接层处理
            fc_output = torch.relu(self.fc1(last_output))  # [batch_size, hidden_size]
            fc_output = self.dropout(fc_output)
            output = self.fc2(fc_output)  # [batch_size, vocab_size]
            
            return output  # [batch_size, vocab_size]


#我想要生成一些随机的片段给他作为训练

def generate_train_text(words, word_to_idx, seq_len=50, num_samples=2000):
    """
    从原始文本中随机选择片段进行训练
    """
    word_indices = [word_to_idx.get(word, word_to_idx.get('<UNK>')) for word in words]
    
    idx_pairs = []
    for i in range(num_samples):
        # 随机选择一个起始位置，确保有足够的位置
        start_idx = random.randint(0, len(word_indices) - seq_len - 1)
        
        # 提取训练序列和目标序列
        train_seq = word_indices[start_idx:start_idx + seq_len]
        target = word_indices[start_idx + seq_len]
        
        idx_pairs.append((train_seq, target))
    
    # 将训练数据转换为张量
    train_data = torch.LongTensor([pair[0] for pair in idx_pairs])
    target_data = torch.LongTensor([pair[1] for pair in idx_pairs])
    
    return train_data, target_data


def train_model(model, words, seq_len, word_to_idx, device, lr=0.001, epochs=50, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    criterion = nn.CrossEntropyLoss()
    
    train_tensor, target_tensor = generate_train_text(
        words=words, 
        word_to_idx=word_to_idx, 
        seq_len=seq_len, 
        num_samples=2000
    )
    
    train_tensor = train_tensor.to(device)
    target_tensor = target_tensor.to(device)
    
    losses = []
    best_loss = float('inf')
    no_improve = 0
    best_model_state = None
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        current_loss = loss.item()
        losses.append(current_loss)
        
        # 更新学习率
        scheduler.step(current_loss)
        
        # 早停检查
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # 每5轮输出一次损失
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"epoch: {epoch} , loss: {current_loss}")
        
        # 如果连续patience轮没有改进，提前结束
        if no_improve >= patience and epoch > 20:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return losses


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


def predict_text(model, words, seq_len, pred_len, word_to_idx, idx_to_word, device, temperature=0.8, top_k=50, top_p=0.9):
    """
    使用改进的采样策略生成文本，包含温度、top-k和top-p(nucleus)采样
    
    Args:
        model: 训练好的模型
        words: 初始文本(单词列表)
        seq_len: 序列长度
        pred_len: 需要预测的文本长度
        word_to_idx: 单词到索引的映射
        idx_to_word: 索引到单词的映射
        device: 计算设备
        temperature: 温度参数，控制随机性(较高=更随机，较低=更确定)
        top_k: 只考虑概率最高的前k个单词
        top_p: 只考虑累积概率达到p的单词(nucleus采样)
    """
    model.eval()  # 设置为评估模式
    
    # 将文本转换为索引
    text_indices = switch_text_index(word_to_idx=word_to_idx, words=words)
    
    # 确保文本长度足够
    if len(text_indices) < seq_len:
        print(f"警告：文本长度({len(text_indices)})小于序列长度({seq_len})，将使用填充")
        # 使用UNK索引填充
        text_indices = [word_to_idx['<UNK>']] * (seq_len - len(text_indices)) + text_indices
    
    # 使用文本的最后seq_len个字符作为初始输入
    input_seq = text_indices[-seq_len:]
    
    # 存储生成的单词
    generated_words = []
    
    # 生成pred_len个单词
    with torch.no_grad():  # 不计算梯度，提高推理速度
        for _ in range(pred_len):
            # 准备输入张量
            input_tensor = torch.LongTensor(input_seq).to(device)
            
            # 获取模型输出
            output = model(input_tensor)
            
            # 使用温度参数调整概率分布
            scaled_logits = output / temperature
            
            # 应用top-k采样：只保留概率最高的k个选项
            if top_k > 0:
                indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][-1]
                scaled_logits[indices_to_remove] = float('-inf')
            
            # 计算概率分布
            probs = torch.nn.functional.softmax(scaled_logits, dim=0)
            
            # 应用top-p(nucleus)采样
            if top_p > 0:
                # 按概率排序
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                
                # 移除概率较小的词，使得累计概率超过阈值
                sorted_indices_to_remove = cumulative_probs > top_p
                # 将首个超过阈值的元素保留
                if len(sorted_indices_to_remove) > 0 and sorted_indices_to_remove[0].item() == 1:
                    sorted_indices_to_remove[0] = False
                
                # 将排除的索引设置成0概率
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0
                
                # 重新归一化概率
                if probs.sum() > 0:
                    probs = probs / probs.sum()
            
            # 使用概率分布进行随机采样
            next_idx = torch.multinomial(probs, 1).item()
            pred_word = idx_to_word.get(next_idx, '<UNK>')
            
            # 将预测的单词添加到生成列表
            generated_words.append(pred_word)
            
            # 更新输入序列：移除第一个单词，添加预测的单词
            input_seq = input_seq[1:] + [next_idx]
    
    # 返回生成的单词列表
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
    print(f"使用设备: {device}")
    
    # 加载文本数据
    file_path = r"I:\Data\Nlp\PrideAndPrejudice.txt"
    text = read_text_file(file_path)
    text = preprocess_text(text)
    words = text.split()
    print(f"文本总词数: {len(words)}")
    
    # 模型参数设置
    seq_len = 50
    num_samples = 5000  # 增加训练样本数
    lr = 0.001
    epochs = 100  # 增加训练轮数
    pred_len = 100  # 生成的文本长度
    
    # 构建词汇表
    vocab_size = 10000
    word_to_idx, idx_to_word = build_vocab(text, max_vocab_size=vocab_size)
    vocab_size = len(idx_to_word)
    print(f"词汇表大小: {vocab_size}")
    
    # 模型超参数
    embedding_dim = 128  # 增加嵌入维度
    hidden_size = 512
    num_layers = 3
    
    # 创建模型
    model = textRNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers).to(device)
    
    # 训练模型
    print("开始训练模型...")
    losses = train_model(
        model,
        words,
        seq_len=seq_len,
        word_to_idx=word_to_idx,
        device=device,
        lr=lr,
        epochs=epochs,
        patience=10)  # 添加早停参数
    
    # 保存模型
    torch.save(model.state_dict(), 'word_level_model.pt')
    print("模型已保存到 word_level_model.pt")
    
    # 尝试不同温度参数生成文本
    print("\n不同参数生成效果对比:")
    
    # 温度低 - 更确定性的输出
    print("\n温度=0.5 (更确定性):")
    seed_text = words[:seq_len]  # 使用原文本开头作为种子
    output_text_low_temp = predict_text(
        model, seed_text, seq_len, pred_len, 
        word_to_idx, idx_to_word, device, 
        temperature=0.5, top_k=40, top_p=0.9
    )
    print(' '.join(output_text_low_temp))
    
    # 平衡温度
    print("\n温度=0.8 (平衡):")
    output_text_mid_temp = predict_text(
        model, seed_text, seq_len, pred_len, 
        word_to_idx, idx_to_word, device, 
        temperature=0.8, top_k=50, top_p=0.9
    )
    print(' '.join(output_text_mid_temp))
    
    # 温度高 - 更随机的输出
    print("\n温度=1.2 (更随机):")
    output_text_high_temp = predict_text(
        model, seed_text, seq_len, pred_len, 
        word_to_idx, idx_to_word, device, 
        temperature=1.2, top_k=0, top_p=0.95
    )
    print(' '.join(output_text_high_temp))
