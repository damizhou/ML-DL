#!/usr/bin/env python3
"""
最小可运行的PyTorch训练脚本 - 适配嵌套目录结构
适用于DFNoDefNet模型和变长流量序列数据
环境要求: torch, numpy, tqdm
"""

# ==================== 1. 导入库 ====================
import torch, sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
# —— 导入你的 DF 模型（保持工程结构）——
ROOT = Path(__file__).resolve().parents[2]  # .../DeepFingerprinting
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Model_NoDef_pytorch import DFNoDefNet  # noqa: E402


# ==================== 2. 配置文件 ====================
CONFIG = {
    'data_dir': '/home/pcz/DL/ML&DL/DeepFingerprinting/DatasetDealer/VPN/npz_longflows',
    'label_mapping': '/home/pcz/DL/ML&DL/DeepFingerprinting/DatasetDealer/VPN/npz_longflows/npz_longflows_labels.json',
    'batch_size': 128,  # 4090推荐128-256
    'learning_rate': 0.001,
    'num_epochs': 30,
    'seq_length': 5000,  # 模型输入固定长度
    'model_save_path': './best_model.pth',
    'num_workers': 4,  # 4090建议4-8
    'mixed_precision': True,  # 4090特有优化
}


# ==================== 3. 数据集类 ====================
class TrafficDataset(Dataset):
    """加载并预处理流量序列数据"""

    def __init__(self, samples, seq_length=5000):
        """
        Args:
            samples: 列表，每个元素是 (flow_array, label_id) 元组
            seq_length: 序列固定长度
        """
        self.samples = samples
        self.seq_length = seq_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flow, label = self.samples[idx]
        # 转换为tensor并添加channel维度 (L,) -> (1, L)
        flow_tensor = torch.from_numpy(flow).float().unsqueeze(0)
        return flow_tensor, torch.tensor(label, dtype=torch.long)


# ==================== 4. 数据加载函数 ====================
def get_data_loaders(config):
    """创建训练、验证和测试DataLoader (按样本8:1:1划分)"""
    data_dir = Path(config['data_dir'])

    # 递归搜索所有子目录中的npz文件
    npz_files = list(data_dir.rglob('*.npz'))

    if not npz_files:
        raise ValueError(f"在 {data_dir} 中未找到npz文件")

    print(f"发现 {len(npz_files)} 个数据文件")

    # 加载标签映射
    with open(config['label_mapping'], 'r') as f:
        label_mapping = json.load(f)
        label2id = label_mapping['label2id']

    # 更新类别数
    config['num_classes'] = len(label2id)

    # 加载所有样本
    all_samples = []
    for file_path in tqdm(npz_files, desc="Loading all samples"):
        try:
            data = np.load(file_path, allow_pickle=True)
            flows = data['flows']  # shape: (N,)
            labels = data['labels']  # shape: (N,)

            for i in range(len(flows)):
                flow = flows[i]  # 变长int8数组
                label_str = str(labels[i])  # 转为字符串

                # 跳过未知标签
                if label_str not in label2id:
                    continue

                # Padding或截断到固定长度
                flow_len = len(flow)
                if flow_len < config['seq_length']:
                    padded = np.zeros(config['seq_length'], dtype=np.int8)
                    padded[:flow_len] = flow
                    flow = padded
                elif flow_len > config['seq_length']:
                    flow = flow[:config['seq_length']]
                else:
                    flow = flow.astype(np.int8)

                all_samples.append((flow, label2id[label_str]))

        except Exception as e:
            print(f"Warning: 加载 {file_path} 失败: {e}")
            continue

    total_samples = len(all_samples)
    print(f"总样本数: {total_samples}")

    if total_samples == 0:
        raise ValueError("未加载到任何有效样本！")

    # 按类别进行8:1:1分层划分
    from collections import defaultdict
    samples_by_class = defaultdict(list)
    for flow, label_id in all_samples:
        samples_by_class[label_id].append((flow, label_id))

    train_samples, val_samples, test_samples = [], [], []

    for class_id, samples in samples_by_class.items():
        n_total = len(samples)
        if n_total < 3:  # 如果某个类别样本太少，全部放入训练集
            train_samples.extend(samples)
            print(f"Warning: 类别 {class_id} 只有 {n_total} 个样本，全部放入训练集")
            continue

        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)

        train_samples.extend(samples[:n_train])
        val_samples.extend(samples[n_train:n_train + n_val])
        test_samples.extend(samples[n_train + n_val:])

    # 检查数据集是否为空
    if not train_samples:
        raise ValueError("训练集为空！请检查数据")
    if not val_samples:
        # 如果验证集为空，从训练集中划分10%作为验证集
        val_split = int(0.1 * len(train_samples))
        val_samples = train_samples[:val_split]
        train_samples = train_samples[val_split:]
        print(f"Warning: 验证集为空，从训练集划分 {val_split} 个样本作为验证集")
    if not test_samples:
        # 如果测试集为空，从剩余训练集中划分10%作为测试集
        test_split = int(0.1 * len(train_samples))
        test_samples = train_samples[:test_split]
        train_samples = train_samples[test_split:]
        print(f"Warning: 测试集为空，从训练集划分 {test_split} 个样本作为测试集")

    print(f"训练样本: {len(train_samples)}, 验证样本: {len(val_samples)}, 测试样本: {len(test_samples)}")

    # 创建数据集
    train_dataset = TrafficDataset(train_samples, config['seq_length'])
    val_dataset = TrafficDataset(val_samples, config['seq_length'])
    test_dataset = TrafficDataset(test_samples, config['seq_length'])

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True)

    return train_loader, val_loader, test_loader, config

# ==================== 5. 训练函数 ====================
def train_epoch(model, train_loader, criterion, optimizer, scaler, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="训练")
    for flows, labels in pbar:
        flows = flows.to(config['device'], non_blocking=True)
        labels = labels.to(config['device'], non_blocking=True)

        optimizer.zero_grad()

        # 混合精度前向传播
        with torch.amp.autocast(enabled=config['mixed_precision'], device_type=config['device'].type):
            outputs = model(flows)
            loss = criterion(outputs, labels)

        # 反向传播
        if config['mixed_precision']:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100. * correct / total:.2f}%'})

    return total_loss / len(train_loader), correct / total


# ==================== 6. 验证函数 ====================
def validate(model, val_loader, criterion, config):
    """验证模型，支持返回预测结果用于计算F1分数"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for flows, labels in tqdm(val_loader, desc="验证"):
            flows = flows.to(config['device'], non_blocking=True)
            labels = labels.to(config['device'], non_blocking=True)

            with torch.amp.autocast(device_type='cuda', enabled=config['mixed_precision']):
                outputs = model(flows)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 收集预测结果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_acc = correct / total if total > 0 else 0

    return avg_loss, avg_acc, all_predictions, all_labels


# ==================== 7. 主函数 ====================
def main():
    """主训练流程"""
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True  # 4090优化

    # 配置
    config = CONFIG.copy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    config['mixed_precision'] = config['mixed_precision'] and (device.type == 'cuda')

    print(f"使用设备: {device}")
    print(f"混合精度: {config['mixed_precision']}")

    # 加载数据 (返回3个loader)
    train_loader, val_loader, test_loader, config = get_data_loaders(config)

    # 创建模型
    model = DFNoDefNet(num_classes=config['num_classes']).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

    # 混合精度缩放器
    scaler = torch.amp.GradScaler() if config['mixed_precision'] else None

    # 训练循环
    best_acc = 0
    for epoch in range(config['num_epochs']):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, config)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, config)

        # 计算 macro F1
        macro_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        print(f"\n训练结果 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"验证结果 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro F1: {macro_f1:.4f}")

        # 学习率调度
        scheduler.step(val_acc)

        # 保存最佳模型 (基于验证准确率)
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'val_acc': val_acc, 'val_f1': macro_f1,
                'config': config, }
            torch.save(checkpoint, config['model_save_path'])
            print(f"✓ 保存最佳模型 (val_acc: {val_acc:.4f}, val_f1: {macro_f1:.4f})")

    # 在测试集上评估最佳模型
    print(f"\n{'=' * 50}")
    print("在测试集上评估最佳模型...")

    # 加载最佳模型
    checkpoint = torch.load(config['model_save_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 测试并获取详细预测结果
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, config)

    # 计算 macro F1
    test_macro_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)

    print(f"测试结果 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Macro F1: {test_macro_f1:.4f}")

    # 打印详细的分类报告
    print("\n详细分类报告:")
    print(classification_report(test_labels, test_preds, digits=4, zero_division=0))

    print(f"\n{'=' * 50}")
    print(f"训练完成! 最佳验证准确率: {best_acc:.4f}, 测试准确率: {test_acc:.4f}, 测试Macro F1: {test_macro_f1:.4f}")

# ==================== 8. 入口 ====================
if __name__ == '__main__':
    main()