# YaTC 实现细节

本文档详细说明了 YaTC 模型的实现细节，包括与原实现的对比、关键实现决策和注意事项。

## 1. 与原实现的对比

### 1.1 依赖变化

| 依赖 | 原版本 | 新版本 | 说明 |
|------|--------|--------|------|
| Python | 3.8 | 3.12 | 语法和标准库更新 |
| PyTorch | 1.9.0 | 2.9.0 | API 更新，性能优化 |
| timm | 0.3.2 | 移除 | 自行实现 VisionTransformer |
| NumPy | 1.19.5 | ≥1.24 | 兼容性更新 |

### 1.2 API 变化处理

#### timm 0.3.2 特有的 `qk_scale` 参数

原实现：
```python
# timm 0.3.2
class Attention:
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, ...):
        self.scale = qk_scale or head_dim ** -0.5
```

新实现：
```python
# 移除 qk_scale，使用标准缩放
class Attention:
    def __init__(self, dim, num_heads=8, qkv_bias=False, ...):
        self.scale = self.head_dim ** -0.5
```

#### PyTorch 2.x 的 meshgrid

原实现：
```python
grid = torch.meshgrid(grid_h, grid_w)  # 无 indexing 参数
```

新实现：
```python
grid = torch.meshgrid(grid_h, grid_w, indexing='ij')  # 显式指定
```

### 1.3 权重兼容性

新实现与原模型的预训练权重**完全兼容**：
- 模型参数名称保持一致
- 参数形状完全相同
- 可直接加载原始预训练权重

## 2. 关键实现决策

### 2.1 Patch 嵌入的特殊处理

YaTC 的 Patch 嵌入与标准 ViT 不同：

```python
# 标准 ViT: 整张图像一次性嵌入
img_size = 40
num_patches = (40/2) * (40/2) = 400

# YaTC: 每个数据包单独嵌入后拼接
img_size = (8, 40)  # 每个数据包
num_patches_per_packet = (8/2) * (40/2) = 80
num_packets = 5
total_patches = 80 * 5 = 400
```

这种设计保留了数据包的层级结构信息。

### 2.2 位置编码扩展

原始 2D 位置编码针对单个数据包生成，然后复制 5 次：

```python
def initialize_weights(self):
    # 生成单个数据包的位置编码
    pos_embed = get_2d_sincos_pos_embed(embed_dim, (4, 20), cls_token=True)

    # 扩展到 5 个数据包
    pos_embed_full = torch.zeros(1, 401, embed_dim)
    pos_embed_full[0, 0] = pos_embed[0]  # CLS token
    for i in range(5):
        pos_embed_full[0, 1 + i * 80: 1 + (i + 1) * 80] = pos_embed[1:]
```

### 2.3 掩码策略

MAE 使用随机掩码，实现细节：

```python
def random_masking(x, mask_ratio):
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))  # 保留 10%

    # 生成随机噪声
    noise = torch.rand(B, N, device=x.device)

    # 排序获取保留的索引
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # 保留前 len_keep 个 patch
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    return x_masked, mask, ids_restore
```

## 3. 训练细节

### 3.1 学习率调度

#### 预训练：步数基准的余弦衰减

```python
def adjust_learning_rate_pretrain(optimizer, step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
```

#### 微调：轮次基准 + 逐层衰减

```python
def get_param_groups_with_layer_decay(model, base_lr, layer_decay, num_layers):
    for name, param in model.named_parameters():
        layer_id = get_layer_id(name, num_layers)
        lr_scale = layer_decay ** (num_layers + 1 - layer_id)
        param_group['lr_scale'] = lr_scale
```

层级学习率：
- 层 0 (patch_embed): lr × 0.65^5 = lr × 0.116
- 层 1 (block 0): lr × 0.65^4 = lr × 0.179
- 层 2 (block 1): lr × 0.65^3 = lr × 0.275
- 层 3 (block 2): lr × 0.65^2 = lr × 0.423
- 层 4 (block 3): lr × 0.65^1 = lr × 0.650
- 层 5 (head): lr × 0.65^0 = lr × 1.000

### 3.2 正则化策略

| 策略 | 预训练 | 微调 |
|------|--------|------|
| Weight Decay | 0.05 | 0.05 |
| Drop Path | 0 | 0.1 |
| Label Smoothing | - | 0.1 |
| Mixup | - | 0 |
| Cutmix | - | 0 |

### 3.3 优化器配置

预训练：
```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.95),
    weight_decay=0.05
)
```

微调：
```python
optimizer = AdamW(
    param_groups,  # 含逐层学习率
    lr=2e-3,
    betas=(0.9, 0.999),
    weight_decay=0.05
)
```

## 4. 数据处理

### 4.1 MFR 生成流程

```
PCAP 文件
    ↓
解析数据包
    ↓
提取前 5 个数据包
    ↓
每个数据包:
├── 头部: 前 80 字节 (填充至 80 字节)
└── 载荷: 前 240 字节 (填充至 240 字节)
    ↓
拼接: 5 × 320 = 1600 字节
    ↓
重塑: 40 × 40 矩阵
    ↓
保存为 PNG 图像
```

### 4.2 数据增强

论文中未使用数据增强，保持原始 MFR 表示。

## 5. 评估指标

```python
def evaluate(model, data_loader, num_classes):
    # 准确率
    accuracy = correct / total

    # 宏平均指标
    for cls in range(num_classes):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

    macro_precision = mean(precisions)
    macro_recall = mean(recalls)
    macro_f1 = mean(f1s)
```

## 6. 内存优化

### 6.1 混合精度训练

虽然本实现未包含，但建议在实际训练时使用：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss, _, _ = model(samples, mask_ratio=0.9)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 6.2 梯度累积

对于大批量训练：

```python
accumulation_steps = 4
for i, samples in enumerate(dataloader):
    loss = model(samples)[0] / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 7. 调试技巧

### 7.1 检查模型输出形状

```python
model = mae_yatc()
x = torch.randn(2, 1, 40, 40)

# 检查编码器输出
latent, mask, ids_restore = model.forward_encoder(x, mask_ratio=0.9)
print(f"Latent shape: {latent.shape}")  # (2, 41, 192)
print(f"Mask shape: {mask.shape}")      # (2, 400)

# 检查解码器输出
pred = model.forward_decoder(latent, ids_restore)
print(f"Pred shape: {pred.shape}")      # (2, 400, 4)
```

### 7.2 验证权重加载

```python
# 加载原始权重
checkpoint = torch.load("original_weights.pth")
model = traformer_yatc(num_classes=7)

# 检查匹配情况
model_keys = set(model.state_dict().keys())
ckpt_keys = set(checkpoint['model'].keys())

missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys

print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")
```

## 8. 常见问题

### Q1: 为什么使用 90% 的掩码比例？

A: 网络流量具有高冗余性，相邻数据包之间存在强相关性。高掩码比例迫使模型学习更深层的语义特征，而不是简单的局部模式。

### Q2: 为什么每个数据包单独嵌入？

A: 这种设计保留了数据包的层级结构信息，使模型能够同时学习包内模式（空间）和包间模式（时序）。

### Q3: 如何处理短于 5 个数据包的流？

A: 使用零填充。如果流少于 5 个数据包，缺失的数据包位置填充为全零。
