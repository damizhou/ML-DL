#!/bin/bash
# FS-Net 消融实验 - 模型训练脚本
#
# 前置条件：
#   已运行 python ablation_processor.py 生成数据文件
#
# 使用方法：
#   chmod +x train_experiments.sh
#   ./train_experiments.sh

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ========================================
# 配置参数 - 根据实际环境修改
# ========================================
WORK_DIR="/home/pcz/DL/ML_DL/FS-Net"
EPOCHS=200          # 训练轮数
BATCH_SIZE=2048     # 批大小 (RTX 4090 推荐)
LEARNING_RATE=0.0005
# ========================================

# 检查工作目录
cd "$WORK_DIR" || {
    print_error "Cannot change to FS-Net directory: $WORK_DIR"
    exit 1
}

print_info "FS-Net Ablation Study - Model Training"
echo "=========================================="
echo ""

# 检查数据文件是否存在
print_info "Checking processed data files..."

if [ ! -f "data/ablation_study/dataset_a_batch.pkl" ]; then
    print_error "Data file not found: data/ablation_study/dataset_a_batch.pkl"
    print_error "Please run 'python ablation_processor.py' first on the data processing device"
    exit 1
fi

if [ ! -f "data/ablation_study/dataset_b_single.pkl" ]; then
    print_error "Data file not found: data/ablation_study/dataset_b_single.pkl"
    print_error "Please run 'python ablation_processor.py' first on the data processing device"
    exit 1
fi

print_info "Data files found!"
echo ""

# 实验1 - 基准线（仅首页）
print_info "Step 1: Running Experiment 1 (Baseline: Homepage Only)..."
print_info "Expected: High accuracy on homepage, low accuracy on sessions (~30%)"

python train_ablation.py \
    --experiment 1 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE || {
    print_error "Experiment 1 failed!"
    exit 1
}

print_info "Experiment 1 completed!"
echo ""

# 实验2 - 提出的方法（全站指纹）
print_info "Step 2: Running Experiment 2 (Proposed: Full-site Fingerprinting)..."
print_info "Expected: High accuracy on sessions (~90%)"
print_info "This is the CORE experiment proving that subpage collection is necessary!"

python train_ablation.py \
    --experiment 2 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE || {
    print_error "Experiment 2 failed!"
    exit 1
}

print_info "Experiment 2 completed!"
echo ""

# 实验3 - 进阶对比（直接会话训练）
print_info "Step 3: Running Experiment 3 (Advanced: Session Training)..."
print_info "Expected: Similar accuracy to Experiment 2 (~90%)"

python train_ablation.py \
    --experiment 3 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE || {
    print_error "Experiment 3 failed!"
    exit 1
}

print_info "Experiment 3 completed!"
echo ""

# 完成
print_info "=========================================="
print_info "All experiments completed successfully!"
print_info "=========================================="
echo ""

print_info "Results saved in:"
echo "  - checkpoints/ablation_study/experiment_1/"
echo "  - checkpoints/ablation_study/experiment_2/"
echo "  - checkpoints/ablation_study/experiment_3/"
echo ""

print_info "To view training logs:"
echo "  tail -f checkpoints/ablation_study/experiment_2/training.log"
echo ""

print_info "Expected key finding:"
echo "  Experiment 1 (Homepage): ~30% accuracy on sessions"
echo "  Experiment 2 (Full-site): ~90% accuracy on sessions ← 3x improvement!"
echo "  Experiment 3 (Session): ~90% accuracy (baseline comparison)"
echo ""
