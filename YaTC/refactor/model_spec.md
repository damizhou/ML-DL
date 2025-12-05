# YaTC 模型规格说明

本文档详细描述了 YaTC 模型的架构规格，所有参数均与论文保持一致。

## 1. 输入格式

### MFR (Multi-level Flow Representation)

```
输入形状: (B, 1, 40, 40)
- B: 批量大小
- 1: 灰度通道
- 40: 高度 (5 packets × 8 rows/packet)
- 40: 宽度 (40 bytes/row)
```

### 数据包结构

```
每个数据包: 320 字节
├── 头部: 80 字节 (160 hex chars)
└── 载荷: 240 字节 (480 hex chars)

每个数据包表示为: 8 × 40 矩阵
每个流包含: 5 个数据包
总计: 5 × 8 × 40 = 1600 字节 → 40 × 40 矩阵
```

## 2. Patch 嵌入

### PatchEmbed 模块

```python
class PatchEmbed:
    img_size = (8, 40)      # 每个数据包的图像大小
    patch_size = (2, 2)     # Patch 大小
    num_patches = 80        # 每个数据包的 Patch 数: (8/2) × (40/2)
    embed_dim = 192         # 嵌入维度
```

### 计算过程

```
1. 每个数据包: (B, 1, 8, 40)
2. Conv2d(1, 192, kernel=2, stride=2): (B, 192, 4, 20)
3. Flatten + Transpose: (B, 80, 192)
4. 5 个数据包拼接: (B, 400, 192)
```

## 3. MAE 预训练模型

### MAE_YaTC 架构

```
输入: (B, 1, 40, 40)
│
├── Patch Embedding
│   └── 输出: (B, 400, 192)
│
├── Position Embedding
│   └── (1, 401, 192) [含 CLS token]
│
├── Random Masking (90%)
│   ├── 保留: (B, 40, 192)
│   └── 掩码: 360 patches
│
├── CLS Token 拼接
│   └── (B, 41, 192)
│
├── Encoder (4 × Transformer Block)
│   ├── Block 0: dim=192, heads=16, mlp_ratio=4
│   ├── Block 1: dim=192, heads=16, mlp_ratio=4
│   ├── Block 2: dim=192, heads=16, mlp_ratio=4
│   └── Block 3: dim=192, heads=16, mlp_ratio=4
│
├── Decoder Embedding
│   └── Linear(192, 128)
│
├── Mask Token 填充
│   └── (B, 401, 128)
│
├── Decoder (2 × Transformer Block)
│   ├── Block 0: dim=128, heads=16, mlp_ratio=4
│   └── Block 1: dim=128, heads=16, mlp_ratio=4
│
└── Prediction Head
    └── Linear(128, 4) → (B, 400, 4)
```

### 编码器参数

| 参数 | 值 | 说明 |
|------|-----|------|
| embed_dim | 192 | 嵌入维度 |
| depth | 4 | Transformer 层数 |
| num_heads | 16 | 注意力头数 |
| head_dim | 12 | 每头维度 (192/16) |
| mlp_ratio | 4 | MLP 隐藏层倍率 |
| mlp_hidden | 768 | MLP 隐藏层维度 (192×4) |
| qkv_bias | True | QKV 偏置 |
| norm_eps | 1e-6 | LayerNorm epsilon |

### 解码器参数

| 参数 | 值 | 说明 |
|------|-----|------|
| decoder_embed_dim | 128 | 解码器嵌入维度 |
| decoder_depth | 2 | 解码器层数 |
| decoder_num_heads | 16 | 解码器注意力头数 |
| decoder_head_dim | 8 | 解码器每头维度 (128/16) |

## 4. TraFormer 微调模型

### TraFormer_YaTC 架构

```
输入: (B, 1, 40, 40)
│
├── Patch Embedding
│   └── 输出: (B, 400, 192)
│
├── CLS Token 拼接
│   └── (B, 401, 192)
│
├── Position Embedding
│   └── (1, 401, 192)
│
├── Dropout (drop_rate)
│
├── Encoder (4 × Transformer Block)
│   ├── Block 0: dim=192, heads=16, drop_path=0.025
│   ├── Block 1: dim=192, heads=16, drop_path=0.050
│   ├── Block 2: dim=192, heads=16, drop_path=0.075
│   └── Block 3: dim=192, heads=16, drop_path=0.100
│
├── LayerNorm
│
├── CLS Token 提取
│   └── (B, 192)
│
└── Classification Head
    └── Linear(192, num_classes) → (B, num_classes)
```

### Drop Path 设置

使用线性增加的 Drop Path Rate:
```python
drop_path_rate = 0.1
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
# dpr = [0.0, 0.033, 0.067, 0.1]
```

## 5. Transformer Block

### Block 结构

```
x → LayerNorm → Attention → DropPath → (+) → LayerNorm → MLP → DropPath → (+) → 输出
│                                      ↑                                    ↑
└──────────────────────────────────────┴────────────────────────────────────┘
                                (残差连接)
```

### Attention 模块

```python
class Attention:
    dim = 192
    num_heads = 16
    head_dim = 12           # dim // num_heads
    scale = 0.2887          # head_dim ** -0.5

    # Q, K, V 投影
    qkv = Linear(192, 576, bias=True)  # 192 * 3 = 576

    # 输出投影
    proj = Linear(192, 192)
```

### MLP 模块

```python
class Mlp:
    in_features = 192
    hidden_features = 768   # 192 * 4
    out_features = 192

    fc1 = Linear(192, 768)
    act = GELU()
    fc2 = Linear(768, 192)
```

## 6. 位置编码

### 2D 正弦余弦位置编码

```python
# 每个数据包的网格大小
grid_size = (4, 20)  # (height/patch_size, width/patch_size)

# 位置编码维度
pos_embed_dim = 192
decoder_pos_embed_dim = 128

# 位置编码形状
pos_embed.shape = (1, 401, 192)       # 400 patches + 1 CLS
decoder_pos_embed.shape = (1, 401, 128)
```

## 7. 权重初始化

### 通用初始化

```python
# Linear 层
nn.init.xavier_uniform_(linear.weight)
nn.init.constant_(linear.bias, 0)

# LayerNorm 层
nn.init.constant_(ln.weight, 1.0)
nn.init.constant_(ln.bias, 0)

# Token
nn.init.normal_(cls_token, std=0.02)
nn.init.normal_(mask_token, std=0.02)
```

### Patch Embedding 初始化

```python
nn.init.xavier_uniform_(patch_embed.proj.weight.view([out_channels, -1]))
```

## 8. 损失函数

### 预训练损失 (MSE)

```python
def forward_loss(imgs, pred, mask):
    target = patchify(imgs)
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # 每个 patch 的平均损失
    loss = (loss * mask).sum() / mask.sum()  # 仅计算掩码 patch
    return loss
```

### 微调损失 (Cross Entropy + Label Smoothing)

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

## 9. 模型参数量

### MAE_YaTC

| 组件 | 参数量 |
|------|--------|
| Patch Embed | ~74K |
| Encoder | ~1.4M |
| Decoder | ~0.5M |
| 总计 | ~2M |

### TraFormer_YaTC

| 组件 | 参数量 |
|------|--------|
| Patch Embed | ~74K |
| Encoder | ~1.4M |
| Head | ~1.3K (7类) |
| 总计 | ~1.5M |

## 10. 预训练权重转移

从 MAE 到 TraFormer 的权重转移：

```python
# 共享的权重
shared_keys = [
    'patch_embed.*',
    'cls_token',
    'pos_embed',
    'blocks.*',
    'norm.*'
]

# 排除的权重 (仅 MAE)
excluded_keys = [
    'decoder_*',
    'mask_token',
    'decoder_pos_embed'
]
```
