#!/bin/bash
# AppScanner 消融实验 - 一键运行脚本
#
# 功能：
# 1. 处理数据集（提取54维统计特征）
# 2. 运行三个消融实验
# 3. 生成结果报告
#
# 使用方法：
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查工作目录
cd /home/pcz/DL/ML_DL/AppScanner || {
    print_error "Cannot change to AppScanner directory"
    exit 1
}

print_info "AppScanner Ablation Study - Automated Experiment Runner"
echo "=========================================="
echo ""

# 配置参数
MODEL="nn"          # 模型类型: nn 或 rf
EPOCHS=100          # 训练轮数
BATCH_SIZE=128      # 批大小

# 步骤1: 检查数据集
print_info "Step 0: Checking dataset..."
if [ ! -d "/netdisk/dataset/ablation_study/batch" ]; then
    print_error "Dataset not found: /netdisk/dataset/ablation_study/batch"
    exit 1
fi
if [ ! -d "/netdisk/dataset/ablation_study/single" ]; then
    print_error "Dataset not found: /netdisk/dataset/ablation_study/single"
    exit 1
fi
print_info "Dataset found!"

# 统计文件数量
BATCH_COUNT=$(find /netdisk/dataset/ablation_study/batch -name "*.pcap" | wc -l)
SINGLE_COUNT=$(find /netdisk/dataset/ablation_study/single -name "*.pcap" | wc -l)
print_info "Batch dataset: $BATCH_COUNT PCAP files"
print_info "Single dataset: $SINGLE_COUNT PCAP files"
echo ""

# 步骤1: 数据处理
print_info "Step 1: Processing datasets (extracting 54-dim features)..."
print_info "This may take 30-60 minutes depending on CPU..."

if [ -f "data/ablation_study/dataset_a_batch.pkl" ] && [ -f "data/ablation_study/dataset_b_single.pkl" ]; then
    print_warn "Processed data already exists. Skip processing? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        python ablation_processor.py || {
            print_error "Data processing failed!"
            exit 1
        }
    else
        print_info "Skipping data processing..."
    fi
else
    python ablation_processor.py || {
        print_error "Data processing failed!"
        exit 1
    }
fi

print_info "Data processing completed!"
echo ""

# 步骤2: 实验1 - 基准线（仅首页）
print_info "Step 2: Running Experiment 1 (Baseline: Homepage Only)..."
print_info "Expected: High accuracy on homepage, low accuracy on sessions (~30%)"

python train_ablation.py \
    --experiment 1 \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE || {
    print_error "Experiment 1 failed!"
    exit 1
}

print_info "Experiment 1 completed!"
echo ""

# 步骤3: 实验2 - 提出的方法（全站指纹）
print_info "Step 3: Running Experiment 2 (Proposed: Full-site Fingerprinting)..."
print_info "Expected: High accuracy on sessions (~90%)"
print_info "This is the CORE experiment proving that subpage collection is necessary!"

python train_ablation.py \
    --experiment 2 \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE || {
    print_error "Experiment 2 failed!"
    exit 1
}

print_info "Experiment 2 completed!"
echo ""

# 步骤4: 实验3 - 进阶对比（直接会话训练）
print_info "Step 4: Running Experiment 3 (Advanced: Session Training)..."
print_info "Expected: Similar accuracy to Experiment 2 (~90%)"

python train_ablation.py \
    --experiment 3 \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE || {
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
echo "  - checkpoints/ablation_study/experiment_1_${MODEL}/"
echo "  - checkpoints/ablation_study/experiment_2_${MODEL}/"
echo "  - checkpoints/ablation_study/experiment_3_${MODEL}/"
echo ""

print_info "To view training logs:"
echo "  tail -f checkpoints/ablation_study/experiment_2_${MODEL}/training.log"
echo ""

print_info "Next steps:"
echo "  1. Compare test accuracies across experiments"
echo "  2. Analyze confusion matrices"
echo "  3. Create plots for paper"
echo ""

print_info "Expected key finding:"
echo "  Experiment 1 (Homepage): ~30% accuracy on sessions"
echo "  Experiment 2 (Full-site): ~90% accuracy on sessions ← 3x improvement!"
echo "  Experiment 3 (Session): ~90% accuracy (baseline comparison)"
