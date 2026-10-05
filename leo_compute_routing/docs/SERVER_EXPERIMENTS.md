# Linux 服务器实验：先固定一个 seed

对应 2026-10-05 的 contact/schema-2 代码。默认固定**训练初始化 2026、独立测试 seed 201**。训练任务回放仍使用 0–19，验证使用 100–104，与测试隔离。单 seed 结果明确标记为预实验，置信区间留空。

优先使用 **compute24（计算密集）** 和 **contact66（接触变化）**。两者采用 600 s 到达窗口、最长 300 s 排空窗口。普通耦合场景此前启发式成功率接近饱和；新方法尚未完成长窗口训练，不能保证每个模块都获得优势。

## 1. 环境

上传当前整个 `leo_compute_routing/` 源码目录，包含 src、scripts、configs、tests、docs、requirements 和 pyproject。当前修改尚未提交，仅克隆旧仓库会遗漏新代码。不要复制 Windows 的 `.conda-env/` 或 Windows wheel 到 Linux。

以下在 **Bash** 执行，第一行改为服务器上的实际目录：

```bash
cd ~/paper2/leo_compute_routing
conda create -n leo-contact python=3.11 pip -y
conda activate leo-contact
python -m pip install -r requirements-dev.txt
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-rl.txt
python -m pip check
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python -m pytest -q
```

安装地址来自 [PyTorch 官方版本页](https://pytorch.org/get-started/previous-versions/)。如果驱动不支持 cu126，可按官方说明改用同版本 cu118 wheel，然后再次验证：

```bash
python -m pip install --force-reinstall --no-deps torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
```

无需 torchvision/torchaudio。没有可用 GPU 时，给训练批量命令加 `--device cpu`。

## 2. 公共变量与功能检查

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
STAMP=$(date +%Y%m%d_%H%M%S)
PILOT="results/server_pilot_${STAMP}"
RUN="results/server_full_${STAMP}"
echo "PILOT=$PILOT"
echo "RUN=$RUN"

# 查看实际命令，不训练、不生成目录
python scripts/run_server_experiments.py --suite main --cases compute24 --updates 20 --output "$PILOT" --dry-run

# 短合成功能检查，不作为论文效果证据
python scripts/run_server_experiments.py --suite main --cases smoke --updates 1 --output "results/server_smoke_${STAMP}"
```

批量入口默认初始化 2026、测试 201、CUDA 训练、CPU 评估，顺序执行。已完成且设置完全一致的训练可复用；每个子命令有独立日志，成功命令记录耗时，失败会停止并保留数据。

## 3. 先跑主对比：20 updates

```bash
python -u scripts/run_server_experiments.py --suite main --cases compute24 --updates 20 --output "$PILOT"
python -u scripts/run_server_experiments.py --suite main --cases contact66 --updates 20 --output "$PILOT"
```

每个场景分别训练完整 GAT 方法与 MLP 对照，每次 update 两个完整 episodes；然后用相同 seed 201 对比两个检查点与七种基线。MLP 只改变节点编码，候选、Shield、资源分配和训练预算保持一致。

完整方法是 Contact Candidate + Contact Shield + GAT-PPO + 批次预测预约 + sqrt 分配。基线为 local、shortest_offload、least_load、computing_aware、computing_aware_future、batch_greedy、batch_greedy_future，直接复用已有实现。

20 updates 用于判断趋势和耗时，不代表收敛。每个 episode 都运行完整到达窗口；验证也运行完整窗口。GPU 利用率可能受 Python 仿真和候选搜索限制。

若需要防止 SSH 断开，可用下面命令**代替**上面的前台执行，不要对同一目录同时启动：

```bash
nohup python -u scripts/run_server_experiments.py --suite main --cases compute24 contact66 --updates 20 --output "$PILOT" > "server_pilot_${STAMP}.log" 2>&1 &
echo "PID=$!"
tail -f "server_pilot_${STAMP}.log"
```

顶层日志显示阶段；训练细节查看：

```bash
tail -f "$PILOT/logs/train/compute24/full/init_2026.log"
```

## 4. 从 20 续训到统一 200 updates

```bash
python -u scripts/run_server_experiments.py --suite main --cases compute24 contact66 --updates 200 --resume-root "$PILOT" --output "$RUN"
```

目标是**总共 200**，不是再加 200。从 last.pt 恢复优化器及随机状态，测试使用验证集选出的 best.pt。续训输出放入新目录，原数据保留；只允许改变设备和目标 updates。

每个模型完整预算为 400 个训练 episodes、标准验证约 100 个 episodes；单 seed 不等于只运行一个 episode。根据首轮日志估计服务器时间，不必第一轮启动全部消融。

如果中途失败，用新输出根目录和原根目录 `--resume-root` 恢复。没有对应源 checkpoint 的任务会提示并从头训练；已经达到目标 updates 的源任务不能再次续训，应复用原目录或仅选未完成 case。某 case 内完成与未完成混合时，使用第 9 节底层命令恢复单个任务。新恢复目录不会自动合并旧目录的其他任务。

## 5. 主要消融

```bash
# Shield：full / no_shield / static_mask
python -u scripts/run_server_experiments.py --suite shield --cases contact66 --updates 200 --output "$RUN"

# 逐个移除接触候选、未来预测、批次预约
python -u scripts/run_server_experiments.py --suite modules --cases contact66 --updates 200 --output "$RUN"

# sqrt / equal，各自在对应的物理分配器下测试
python -u scripts/run_server_experiments.py --suite resource --cases compute24 --updates 200 --output "$RUN"
```

以上复用已完成的 full，其余变体独立训练，测试固定 201。

| suite | 比较方法 | 主要指标 |
| --- | --- | --- |
| main | full / mlp + 七种基线 | 成功率、P95、任务成本、实际失败 |
| shield | full / no_shield / static_mask | 训练期预测违规、fallback、实际违约与断链 |
| modules | full / ksp / no_future / no_booking | 候选、预测、批次竞争各自的贡献 |
| resource | full / equal | 局部资源分配的作用 |

no_shield 保留相同候选与预测特征，只关闭屏蔽；static_mask 使用原静态 mask。resource 输出在 `eval/resource_own/{full,equal}/`，避免将 equal 检查点误放到 sqrt 环境。

算力充足时可统一补齐两个主场景八种变体，共 16 次独立训练；它包含主对比与消融全部变体，不包含跨规模与敏感性：

```bash
python -u scripts/run_server_experiments.py --suite all --cases compute24 contact66 --updates 200 --output "$RUN"
```

第一轮推荐 main → shield → modules；all 很耗时。相同目录不可并发写入。

## 6. 跨规模零样本测试

```bash
python -u scripts/run_server_experiments.py --suite scale --cases scale48 --updates 200 --output "$RUN"
```

在 48 星分别训练 full/MLP，冻结权重与训练尺度后测试 24/48/72/96 星，均使用 seed 201，保留 48 星 held-out 参照。总到达率按星数同比缩放，默认包含七种基线。

若先节约评估计算，可在首次运行时给 scale 命令加 `--algorithms local computing_aware batch_greedy`；已完成同名评估后不能换参数覆盖。cost shift 同时包含目标场景固有难度，不是纯泛化误差。

## 7. 补充场景、审计、敏感性

```bash
# 通信受限、普通耦合的补充主对比
python -u scripts/run_server_experiments.py --suite main --cases link24 coupled24 --updates 200 --output "$RUN"

# 物理审计使用诊断 seed=50，不与测试 201 混用
python -u scripts/run_server_experiments.py --suite audit --cases compute24 contact66 scale48 --output "$RUN"

# 冻结已训练 full/MLP 的鲁棒性扫描，不重新选模型
python -u scripts/run_server_experiments.py --suite sensitivity --phase evaluate --cases compute24 --output "$RUN"
```

敏感性共 15 点，每次只改一个量：到达率 ×[0.5,1,1.5]、CPU ×[0.5,1,1.5]、任务业务链路容量 ×[0.5,1,2]、deadline ×[0.75,1,1.25]、lookahead=8/16/32 slots（名义 2/4/8 s）。完整保存所有点，包括各组重复的基准点。

这是冻结策略鲁棒性，不能称为每点重训练后的最优性能。严苛 deadline / 低 CPU 点可能先天不可行，需同时看对应 calibration。若要研究各参数下重训练性能，另建配置和独立目录。

## 8. 输出与解释

```text
results/server_full_时间戳/
├── train/compute24/full/init_2026/
│   ├── best.pt / last.pt / resolved_config.yaml
│   ├── episodes.csv / validation.csv / updates.csv
│   ├── training_summary.json / training_manifest.json
│   └── learning_curve.png / shield_curve.png
├── eval/main/compute24/init_2026/
│   ├── seed_metrics.csv / aggregate.csv / paired_differences.csv
│   ├── evaluation_study.json / variant_mapping.json
│   └── seed_201/                 原始任务记录、summary、calibration
├── eval/shield/、eval/modules/、eval/resource_own/...
├── generalization/init_2026/     四规模、held-out 参照、cost_shift.csv
├── sensitivity_configs/         扫描实际配置
├── logs/                        每个子命令日志
└── runner_status/               成功命令及耗时
```

先看 validation 是否仍改善，再看测试 success_rate、mean_completion_delay_s、p95_completion_delay_s、route_failure_rate、deadline_violation_rate、censored_rate 和 mean_cost_per_admitted_task_s。平均时延仅包含完成任务，不能只看这一列。Shield 的 predicted_invalid_action_rate 是预测不合规选择，不是实际断链；预测预约不是物理服务保证。

单 seed 的 CI 留空。variant_mapping.json 明确 gat_ppo_2 等名称对应哪个消融，不要凭后缀猜。接触场景历史抽样风险约 0.5%–0.7%，future 实际贡献可能较小，仍需以选择路径和实际结果为准。

入口只复用设置一致的已完成训练，以及相同命令/检查点的已完成评估，失败输出不静默覆盖。续训的新目录仅记录本次续训日志；绘制完整 1–200 曲线时需按 update 合并源与新日志。

## 9. 底层命令

不用批量入口时：

```bash
python -u scripts/train_ppo.py --config configs/experiments/contact_compute_heavy_ppo.yaml --set rl.seed=2026 --set rl.device=cuda --set rl.updates=200 --output results/manual_compute_gat
python -u scripts/train_ppo.py --config configs/experiments/contact_compute_heavy_ppo.yaml --set rl.seed=2026 --set rl.encoder=mlp --set rl.device=cuda --set rl.updates=200 --output results/manual_compute_mlp
python -u scripts/evaluate_ppo.py --config configs/experiments/contact_compute_heavy_ppo.yaml --checkpoints results/manual_compute_gat/best.pt results/manual_compute_mlp/best.pt --seeds 201 --algorithms local shortest_offload least_load computing_aware computing_aware_future batch_greedy batch_greedy_future --device cpu --output results/manual_compute_test201
python scripts/plot_training.py results/manual_compute_gat
```

开关：no_shield=`--set rl.shield_mode=none`，static_mask=`--set rl.shield_mode=mask`，ksp=`--set routing.candidate_generation=ksp`，no_future=`--set rl.use_future=false`，no_booking=`--set rl.use_reservations=false`，equal=`--set resource.allocation=equal`。每个变体独立训练并新建目录。

恢复单个任务时加 `--resume 原目录/last.pt`，复写原训练参数，输出新目录；不用 best.pt 恢复。

## 10. 后续多 seed

目前先做单初始化。确认趋势后在新结果根目录扩展：

```bash
python -u scripts/run_server_experiments.py --suite main --cases compute24 contact66 --initializations 2026 2027 2028 --test-seeds 201 202 203 204 205 --updates 200 --output "results/server_multiseed_$(date +%Y%m%d_%H%M%S)"
```

每个初始化内的 CI 仅覆盖测试 seed 变异；最终还需考虑训练初始化变异，不能把全部任务或 3×5 组合当成完全独立样本。

本次入口已通过 main 双模型单 seed、同目录复用、续训与资源分配检查；回归 99 项通过。这些是合成功能检查，不是长窗口性能证据。
