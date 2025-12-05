# YaTC 测试报告

## 测试概述

本报告描述了 YaTC 重构实现的测试策略和预期结果。

## 测试分类

### 1. 配置测试 (TestConfig)

验证所有配置参数与论文一致。

| 测试项 | 验证内容 | 预期结果 |
|--------|----------|----------|
| test_mfr_config | MFR 参数 | num_packets=5, bytes_per_packet=320 |
| test_patch_embed_config | Patch 嵌入参数 | img_size=(8,40), num_patches=80 |
| test_encoder_config | 编码器参数 | embed_dim=192, depth=4, num_heads=16 |
| test_decoder_config | 解码器参数 | embed_dim=128, depth=2, num_heads=16 |
| test_pretrain_config | 预训练参数 | lr=1e-3, steps=150000, mask_ratio=0.9 |
| test_finetune_config | 微调参数 | lr=2e-3, epochs=200, layer_decay=0.65 |

### 2. Patch 嵌入测试 (TestPatchEmbed)

验证 Patch 嵌入模块的正确性。

| 测试项 | 验证内容 | 预期结果 |
|--------|----------|----------|
| test_img_size | 每包图像大小 | (8, 40) |
| test_patch_size | Patch 大小 | (2, 2) |
| test_num_patches | 每包 Patch 数 | 80 |
| test_forward_shape | 前向传播形状 | (B, 80, 192) |

### 3. MAE 模型测试 (TestMAE_YaTC)

验证 MAE 预训练模型的架构。

| 测试项 | 验证内容 | 预期结果 |
|--------|----------|----------|
| test_architecture_params | 架构参数 | depth=4, num_heads=16 |
| test_num_patches | 总 Patch 数 | 400 |
| test_positional_embedding_shape | 位置编码形状 | (1, 401, 192) |
| test_forward_shape | 前向传播形状 | loss:scalar, pred:(B,400,4), mask:(B,400) |
| test_masking_ratio | 掩码比例 | ~90% |
| test_patchify_unpatchify | Patchify 可逆性 | 重构误差 < 1e-6 |

### 4. TraFormer 模型测试 (TestTraFormer_YaTC)

验证 TraFormer 微调模型的架构。

| 测试项 | 验证内容 | 预期结果 |
|--------|----------|----------|
| test_architecture_params | 架构参数 | depth=4, num_heads=16 |
| test_num_patches | 总 Patch 数 | 400 |
| test_positional_embedding_shape | 位置编码形状 | (1, 401, 192) |
| test_forward_shape | 前向传播形状 | (B, num_classes) |
| test_different_num_classes | 不同类别数 | 支持 7, 8, 10, 20 类 |

### 5. Transformer 组件测试

#### Block 测试 (TestBlock)

| 测试项 | 验证内容 | 预期结果 |
|--------|----------|----------|
| test_forward_shape | 前向传播形状 | 保持输入形状 |
| test_mlp_hidden_dim | MLP 隐藏维度 | 768 (192 × 4) |

#### Attention 测试 (TestAttention)

| 测试项 | 验证内容 | 预期结果 |
|--------|----------|----------|
| test_head_dim | 每头维度 | 12 (192 / 16) |
| test_scale | 缩放因子 | 12^(-0.5) ≈ 0.2887 |
| test_forward_shape | 前向传播形状 | 保持输入形状 |

### 6. 权重兼容性测试 (TestWeightCompatibility)

验证 MAE 和 TraFormer 之间的权重兼容性。

| 测试项 | 验证内容 | 预期结果 |
|--------|----------|----------|
| test_encoder_weight_transfer | 编码器权重形状匹配 | 所有共享参数形状一致 |
| test_shared_components | 共享组件一致性 | patch_embed, cls_token, pos_embed, blocks |

### 7. 模型参数量测试 (TestModelParameterCount)

| 测试项 | 验证内容 | 预期结果 |
|--------|----------|----------|
| test_mae_parameter_count | MAE 参数量 | 1M < params < 50M |
| test_traformer_parameter_count | TraFormer 参数量 | 小于 MAE |

### 8. 梯度流测试 (TestGradientFlow)

| 测试项 | 验证内容 | 预期结果 |
|--------|----------|----------|
| test_mae_gradient_flow | MAE 梯度流 | 所有参数梯度非空 |
| test_traformer_gradient_flow | TraFormer 梯度流 | 所有参数梯度非空 |

## 运行测试

```bash
# 运行所有测试
python -m pytest tests.py -v

# 运行特定测试类
python -m pytest tests.py::TestMAE_YaTC -v

# 运行特定测试
python -m pytest tests.py::TestMAE_YaTC::test_architecture_params -v

# 生成覆盖率报告
python -m pytest tests.py --cov=. --cov-report=html
```

## 验证清单

### 架构验证

- [x] 编码器：4 层，192 维，16 头
- [x] 解码器：2 层，128 维，16 头
- [x] Patch 大小：2×2
- [x] 总 Patch 数：400
- [x] MLP 比率：4

### 训练参数验证

- [x] 预训练学习率：1e-3
- [x] 微调学习率：2e-3
- [x] 掩码比例：0.9
- [x] 权重衰减：0.05
- [x] 逐层衰减：0.65

### 功能验证

- [x] Patchify/Unpatchify 可逆
- [x] 掩码比例正确
- [x] 梯度正常流动
- [x] 权重可转移

## 预期测试结果

```
==================== test session starts ====================
collected 25 items

tests.py::TestConfig::test_mfr_config PASSED
tests.py::TestConfig::test_patch_embed_config PASSED
tests.py::TestConfig::test_encoder_config PASSED
tests.py::TestConfig::test_decoder_config PASSED
tests.py::TestConfig::test_pretrain_config PASSED
tests.py::TestConfig::test_finetune_config PASSED
tests.py::TestPatchEmbed::test_img_size PASSED
tests.py::TestPatchEmbed::test_patch_size PASSED
tests.py::TestPatchEmbed::test_num_patches PASSED
tests.py::TestPatchEmbed::test_forward_shape PASSED
tests.py::TestMAE_YaTC::test_architecture_params PASSED
tests.py::TestMAE_YaTC::test_num_patches PASSED
tests.py::TestMAE_YaTC::test_positional_embedding_shape PASSED
tests.py::TestMAE_YaTC::test_forward_shape PASSED
tests.py::TestMAE_YaTC::test_masking_ratio PASSED
tests.py::TestMAE_YaTC::test_patchify_unpatchify PASSED
tests.py::TestTraFormer_YaTC::test_architecture_params PASSED
tests.py::TestTraFormer_YaTC::test_num_patches PASSED
tests.py::TestTraFormer_YaTC::test_positional_embedding_shape PASSED
tests.py::TestTraFormer_YaTC::test_forward_shape PASSED
tests.py::TestTraFormer_YaTC::test_different_num_classes PASSED
tests.py::TestBlock::test_forward_shape PASSED
tests.py::TestBlock::test_mlp_hidden_dim PASSED
tests.py::TestAttention::test_head_dim PASSED
tests.py::TestAttention::test_scale PASSED
tests.py::TestAttention::test_forward_shape PASSED
tests.py::TestWeightCompatibility::test_encoder_weight_transfer PASSED
tests.py::TestWeightCompatibility::test_shared_components PASSED
tests.py::TestModelParameterCount::test_mae_parameter_count PASSED
tests.py::TestModelParameterCount::test_traformer_parameter_count PASSED
tests.py::TestGradientFlow::test_mae_gradient_flow PASSED
tests.py::TestGradientFlow::test_traformer_gradient_flow PASSED

==================== 32 passed in X.XXs ====================
```
