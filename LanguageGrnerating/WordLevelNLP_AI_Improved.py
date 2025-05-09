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
import torch.nn.functional as F
from tqdm import tqdm
import re
from collections import Counter
import random


class textRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, input_size=1, hidden_size=512, output_size=1, num_layers=3, dropout=0.4, context_size=5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 词嵌入和位置编码
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_size, embedding_dim)
        
        # 双向LSTM用于捕捉长期依赖
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # 语法感知层
        self.grammar_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 添加注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 添加层标准化
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)
        
        # 添加全连接层
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        # 检查输入是否为批量输入
        if len(x.shape) == 1:  # 非批量输入 [seq_len]
            x = x.unsqueeze(0)  # [1, seq_len]
        
        batch_size, seq_len = x.size()
        
        # 添加位置编码
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.position_embedding(positions % self.position_embedding.num_embeddings)
        
        # 组合词嵌入和位置编码
        embedded = self.embedding(x) + pos_embeddings
        
        # 初始化LSTM隐藏状态
        h0 = torch.zeros(
            self.num_layers * 2,  # 双向所以是2倍
            batch_size,
            self.hidden_size,
            device=x.device)
        c0 = torch.zeros(
            self.num_layers * 2,
            batch_size,
            self.hidden_size,
            device=x.device)
        
        # LSTM处理
        lstm_out, _ = self.lstm(embedded, (h0, c0))  # [batch_size, seq_len, hidden_size*2]
        
        # 应用语法感知层
        grammar_aware = self.grammar_layer(lstm_out)
        
        # 应用自注意力机制
        attn_output, _ = self.attention(grammar_aware, grammar_aware, grammar_aware)
        
        # 残差连接和层标准化
        output = self.layer_norm1(grammar_aware + attn_output)
        
        # 全连接层
        output = self.fc1(output)  # [batch_size, seq_len, hidden_size*2]
        output = F.gelu(output)  # 使用GELU激活函数
        output = self.dropout(output)
        output = self.fc2(output)  # [batch_size, seq_len, vocab_size]
        
        if len(x.shape) == 1:
            # 只取最后一个时间步的输出
            return output[0, -1]  # [vocab_size]
        else:
            # 取所有批次的最后一个时间步
            return output[:, -1]  # [batch_size, vocab_size]


#我想要生成一些随机的片段给他作为训练

def generate_train_text(words, word_to_idx, seq_len=50, num_samples=2000):
    """
    从原始文本中随机选择片段进行训练
    """
    print(f"开始生成训练数据，样本数: {num_samples}")
    word_indices = [word_to_idx.get(word, word_to_idx.get('<UNK>')) for word in words]
    print("单词转换为索引完成")
    
    idx_pairs = []
    for i in range(num_samples):
        if i % 500 == 0:  # 每500个样本显示一次进度
            print(f"生成训练样本进度: {i}/{num_samples}")
        # 随机选择一个起始位置，确保有足够的位置
        start_idx = random.randint(0, len(word_indices) - seq_len - 1)
        
        # 提取训练序列和目标序列
        train_seq = word_indices[start_idx:start_idx + seq_len]
        target = word_indices[start_idx + seq_len]
        
        idx_pairs.append((train_seq, target))
    
    print("开始转换为张量...")
    # 将训练数据转换为张量
    train_data = torch.LongTensor([pair[0] for pair in idx_pairs])
    target_data = torch.LongTensor([pair[1] for pair in idx_pairs])
    print("数据准备完成")
    
    return train_data, target_data


def evaluate_model(model, val_data, seq_len, word_to_idx, device, criterion):
    """评估模型
    
    Args:
        model: 模型实例
        val_data: 验证数据
        seq_len: 序列长度
        word_to_idx: 词到索引的映射
        device: 设备（CPU/GPU）
        criterion: 损失函数
    
    Returns:
        float: 验证集上的平均损失
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # 创建验证集数据加载器
    val_dataset = torch.utils.data.TensorDataset(val_data)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device)
            targets = inputs[:, 1:].contiguous()
            inputs = inputs[:, :-1].contiguous()
            
            # 前向传播
            output = model(inputs)
            
            # 计算损失
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_model(model, train_data, seq_len, word_to_idx, device, optimizer, scheduler, criterion, epochs=100, patience=5):
    """训练模型
    
    Args:
        model: 模型实例
        train_data: 训练数据
        seq_len: 序列长度
        word_to_idx: 词到索引的映射
        device: 设备（CPU/GPU）
        optimizer: 优化器
        scheduler: 学习率调度器
        criterion: 损失函数
        epochs: 训练轮数
        patience: 早停耐心值
    
    Returns:
        losses: 训练过程中的损失值列表
    """
    # 创建数据加载器进行批次训练
    dataset = torch.utils.data.TensorDataset(train_data, target_data)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )
    
    losses = []
    best_loss = float('inf')
    no_improve = 0
    best_model_state = None
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # 创建进度条
        pbar = tqdm(total=len(train_data) // 32, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_inputs, batch_targets in train_loader:
            # 将数据移到设备上
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            output = model(batch_inputs)
            loss = criterion(output, batch_targets)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            # 清理GPU内存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 计算平均损失
        current_loss = total_loss / num_batches
        losses.append(current_loss)
        
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
    text = text.lower()         # 转小写
    text = text.replace('\n',' ')# 把回车变成空格
    # 只过滤掉真正的特殊字符，保留更多有意义的标点和符号
    text = re.sub(r'[^\w\s.,;\-\'\"?!]',' ',text)
    text = re.sub(r'\s+',' ',text)# 把多个空格变成一个
    return text.strip()


def build_vocab(text, max_vocab_size = 30000):
    print(f"开始构建词汇表，最大大小: {max_vocab_size}")
    words = text.split()
    print(f"文本分词完成，共 {len(words)} 个词")
    word_count = Counter(words)
    common_words = word_count.most_common(max_vocab_size-2)
    print(f"找到 {len(common_words)} 个常用词")
    
    idx_to_word = {0:'<PAD>',1:'<UNK>'}
    word_to_idx = {'<PAD>':0,'<UNK>':1}
    for i,(word,_) in enumerate(common_words):
        idx = i+2
        idx_to_word[idx] = word
        word_to_idx[word] = idx
    
    print(f"词汇表构建完成，大小: {len(word_to_idx)}")
    return word_to_idx,idx_to_word


def switch_text_index(word_to_idx,words):
    text_index = [word_to_idx.get(word,word_to_idx.get('<UNK>',0))for word in words]
    return text_index


def switch_index_text(idx_to_word,output_tensor):
    idx = torch.argmax(output_tensor).item()
    output_ch = idx_to_word.get(idx,idx_to_word.get('<UNK>',0))
    return output_ch


def predict_text(model, words, seq_len, pred_len, word_to_idx, idx_to_word, device, base_temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.2, min_tokens_to_keep=5, dynamic_temp=True):
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
        base_temperature: 基础温度参数，控制随机性(较高=更随机，较低=更确定)
        top_k: 只考虑概率最高的前k个单词
        top_p: 只考虑累积概率达到p的单词(nucleus采样)
        repetition_penalty: 重复惩罚因子，控制生成文本的重复度
        min_tokens_to_keep: 在top-k/top-p采样中要保留的最小标记数
        dynamic_temp: 是否使用动态温度调整
    """
    model.eval()  # 设置为评估模式
    
    # 将文本转换为索引
    text_indices = switch_text_index(word_to_idx=word_to_idx, words=words)
    
    # 确保文本长度足够
    if len(text_indices) < seq_len:
        print(f"警告：文本长度({len(text_indices)})小于序列长度({seq_len})，将使用填充")
        # 使用UNK索引填充
        text_indices = [word_to_idx['<UNK>']] * (seq_len - len(text_indices)) + text_indices
    
    # 将输入单词转换为索引
    input_indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
    
    # 使用最后seq_len个索引作为初始输入
    input_seq = input_indices[-seq_len:]
    
    # 初始化生成的单词列表和索引列表
    generated_words = []
    generated_indices = []
    
    # 生成pred_len个单词
    with torch.no_grad():  # 不计算梯度，提高推理速度
        for _ in range(pred_len):
            # 准备输入张量
            input_tensor = torch.LongTensor(input_seq).to(device)
            
            # 获取模型输出
            output = model(input_tensor)
            
            # 动态调整温度
            if dynamic_temp:
                # 根据生成的文本长度动态调整温度
                # 随着生成越来越多的文本，逐渐降低温度以增加连贯性
                progress = len(generated_words) / pred_len
                current_temp = base_temperature * (1.0 + 0.2 * (1.0 - progress))
            else:
                current_temp = base_temperature
            
            # 应用温度系数
            logits = output / current_temp
            for i in range(len(generated_indices)):
                logits[generated_indices[i]] /= repetition_penalty
            
            # 使用softmax获取概率分布
            probs = torch.softmax(logits, dim=-1)
            
            # 改进的top-k采样：确保至少保留min_tokens_to_keep个词
            if top_k > 0:
                # 确保至少保留min_tokens_to_keep个词
                effective_top_k = min(top_k, probs.size(-1))
                effective_top_k = max(effective_top_k, min_tokens_to_keep)
                
                # 获取前K个概率最高的词
                _, top_indices = torch.topk(probs, effective_top_k)
                
                # 创建一个新的概率分布，只保留top-k的概率
                filtered_probs = torch.zeros_like(probs)
                filtered_probs.scatter_(0, top_indices, probs[top_indices])
                
                # 使用新的概率分布
                probs = filtered_probs
                
                # 对剩余的概率重新归一化
                if probs.sum() > 0:
                    probs = probs / probs.sum()
            
            # 确保概率分布归一化
            if probs.sum() > 0:
                probs = probs / probs.sum()
            
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
            generated_indices.append(next_idx)
            
            # 更新输入序列：移除第一个单词，添加预测的单词
            input_seq = input_seq[1:] + [next_idx]
    
    # 返回生成的单词列表
    return generated_words


def read_text_file(file_path, max_chars=1000000):  # 默认读取前1MB数据用于测试
    print(f"开始读取文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read(max_chars)  # 只读取指定大小的数据
        print(f"文件读取成功，大小: {len(text)} 字符")
        return text
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        raise
    except Exception as e:
        print(f"错误：读取文件时出错 - {str(e)}")
        raise

if  __name__ == "__main__":
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载文本数据
    file_path = r"D:\tbw\data\text8.txt"
    print("\n=== 开始加载数据 ===")
    text = read_text_file(file_path, max_chars=1000000)  # 先用1MB数据测试
    print("\n=== 开始预处理文本 ===")
    text = preprocess_text(text)
    words = text.split()
    print(f"文本总词数: {len(words)}")
    
    # 模型参数设置
    print("\n=== 设置模型参数 ===")
    seq_len = 50
    num_samples = 10000  # 增加训练样本数
    lr = 0.001
    epochs = 100  # 增加训练轮数
    pred_len = 100  # 生成的文本长度
    
    # 构建词汇表
    vocab_size = 30000  # 增加词汇表大小，与build_vocab函数保持一致
    word_to_idx, idx_to_word = build_vocab(text, max_vocab_size=vocab_size)
    vocab_size = len(idx_to_word)
    print(f"词汇表大小: {vocab_size}")
    
    # 训练参数
    num_epochs = 150     # 训练轮数
    batch_size = 64      # 批量大小
    learning_rate = 0.0005  # 学习率
    
    # 模型超参数
    embedding_dim = 512   # 嵌入维度
    hidden_size = 1024   # 隐藏层大小
    num_layers = 4       # LSTM层数
    context_size = 10    # 上下文窗口大小
    
    # 创建模型
    model = textRNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        context_size=context_size
    ).to(device)

    # 使用AdamW优化器和余弦退火学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 使用带标签平滑的交叉熵损失
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 准备训练和验证数据
    print("准备训练数据...")
    train_data, target_data = generate_train_text(
        words=words,
        word_to_idx=word_to_idx,
        seq_len=seq_len,
        num_samples=3000  # 增加训练样本数量
    )
    
    # 划分训练集和验证集
    val_size = int(len(train_data) * 0.1)  # 10%的数据用于验证
    train_data, val_data = train_data[val_size:], train_data[:val_size]
    target_data = target_data[val_size:]
    
    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")
    
    # 训练模型
    print("开始训练模型...")
    losses = train_model(
        model=model,
        train_data=train_data,
        seq_len=seq_len,
        word_to_idx=word_to_idx,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=num_epochs,
        patience=10)
    
    # 保存模型
    torch.save(model.state_dict(), 'word_level_model.pt')
    print("模型已保存到 word_level_model.pt")
    
    # 测试生成文本
    print("生成文本测试:")

    # 分别测试不同温度的生成效果
    test_temps = [0.6, 0.8, 1.0]
    temp_names = ["较保守", "平衡", "更创意"]
    test_k = [20, 30, 40]
    test_p = [0.85, 0.9, 0.92]
    test_penalty = [1.5, 1.8, 2.0]
    
    for i, (temp, name, k, p, penalty) in enumerate(zip(test_temps, temp_names, test_k, test_p, test_penalty)):
        print(f"\n温度={temp} ({name}):")
        generated_text = predict_text(
            model=model,
            words=words[-seq_len:],
            seq_len=seq_len,
            pred_len=100,
            word_to_idx=word_to_idx,
            idx_to_word=idx_to_word,
            device=device,
            base_temperature=temp,
            top_k=k,
            top_p=p,
            repetition_penalty=penalty,
            min_tokens_to_keep=5
        )
        print(' '.join(generated_text))
