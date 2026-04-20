# CODEX Task Queue

- [x] 已完成并验证：统一 AppScanner RF train/val/test 默认 `trees_per_batch` 为 25 - 2026-04-20 09:35

- [x] 已完成并验证：修复 `AppScanner/train_with_dataset.py` 中 train/val/test 随机森林评估参数不一致问题 - 2026-04-20 09:32

- [x] 已完成并验证：生成仓库贡献指南 `AGENTS.md` - 2026-04-18 19:26
- [x] 已完成并验证：核对 `5实验设置与结果分析.md` 与实验代码是否一致 - 2026-04-18 19:41
- [x] 已完成并验证：将第 5 章一致性核查与新增问题答复写入文件 - 2026-04-18 19:52
- [x] 已完成并验证：按用户限定修订第 5 章核查报告，仅聚焦 5.1 方法设置 - 2026-04-18 19:59
- [x] 已完成并验证：核查 AppScanner dataset_a_batch.pkl 生成链路并修订 5.1 报告 - 2026-04-18 20:17
- [-] 修正 AppScanner 随机森林消融实验的训练集指标缺失与 Macro-F1 口径错误；待修复 `dataset_a_batch.pkl` 后重跑表 5-2（高，1.5h）
- [x] 已完成并验证：参考 `DeepFingerprinting/train_multi_datasets.py` 新增 `AppScanner/train_multi_datasets.py` - 2026-04-19 10:52
- [x] 已完成并验证：核查当前实验是否存在目标域系统性超参数搜索缺失与公平性偏差 - 2026-04-19 15:16
- [x] 已完成并验证：为 `AppScanner/train_multi_datasets.py` 增加启动前等待指定 PID 结束的阻塞逻辑 - 2026-04-19 20:47
- [x] 已完成并验证：修复 AppScanner RF `tree_first` 在大样本多类别训练集评估时申请全量概率矩阵导致 OOM - 2026-04-20 15:41
- [x] 已完成并验证：按 512GB 内存与 112 核 CPU 调整 AppScanner RF 训练/评估并行与内存预算 - 2026-04-20 15:59
- [x] 已完成并验证：将 AppScanner RF 默认并行度调整为 100 棵树的整除因子以减少尾批低利用率 - 2026-04-20 17:12
