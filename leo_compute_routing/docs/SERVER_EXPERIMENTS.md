# 服务器实验命令

在 `leo_compute_routing/` 下执行。新代码的 PPO 只选择计算卫星，旧 checkpoint 不可续训；使用一个全新的输出根目录。训练初始化及任务回放都固定为 2026，验证与当前开发比较固定为 100。服务器默认顺序运行，不会自动启动多 seed 训练。

## 1. 环境检查与功能检查

```bash
conda activate leo-paper2
python -c "import torch; print(torch.__version__); print('CUDA_available=', torch.cuda.is_available())"
python -m pytest -q
python -u scripts/train_ppo.py --config configs/rl_smoke.yaml \
  --set rl.device=auto --output "results/smoke_$(date +%Y%m%d_%H%M%S)"
```

需要创建环境时见 [ENVIRONMENT.md](ENVIRONMENT.md)。现有 PyTorch 环境可继续使用，架构重构不增加新依赖。模拟器和图搜索运行于 CPU，网络前向/反向可使用 GPU。`--device auto` 自动选择；需强制 GPU 时用服务器入口 `--device cuda` 或单训练入口 `--set rl.device=cuda`。

## 2. 第一轮主实验：重新训练 40 updates

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
RUN="results/hierarchical_${STAMP}"
nohup python -u scripts/run_server_experiments.py \
  --suite main --phase both --cases compute24 contact66 \
  --updates 40 --device auto --output "$RUN" \
  > "hierarchical_${STAMP}.log" 2>&1 &
tail -f "hierarchical_${STAMP}.log"
```

`main` 会在两个场景分别独立训练 GAT-PPO 和 MLP-PPO，并在 seed 100 比较 local、shortest_offload、least_load、computing_aware、computing_aware_future、batch_greedy、node_greedy。后者与提出的方法共用接触路由、预约及资源，是区分节点长期学习收益的必要对照。40 updates 是预实验，不预先认定已收敛。

如果只想先训练，将 `--phase both` 换成 `--phase train`；结束后对同一个 `$RUN` 执行：

```bash
python -u scripts/run_server_experiments.py --suite main --phase evaluate \
  --cases compute24 contact66 --updates 40 --output "$RUN"
```

控制台包含 update/episode/slot 进度、FPS、ETA、设备实际使用情况、reward、平均/P95 时延、成功率、期限违约、实际路由失败、路由拒绝、截尾、任务代价、CPU/链路利用率以及 PPO loss/KL/entropy/梯度/价值误差。

## 3. 继续到 200 updates

仅能从**新版** `last.pt` 继续；保持场景和学习参数相同，把总 update 数加大，写入新目录：

```bash
MORE="results/hierarchical_200_$(date +%Y%m%d_%H%M%S)"
python -u scripts/run_server_experiments.py --suite main --phase both \
  --cases compute24 contact66 --updates 200 --resume-root "$RUN" --output "$MORE"
```

## 4. 路由、节点决策与资源消融

各命令会训练自己的完整方法和变体，使用同一初始化、回放任务与验证协议。每组新建输出根目录，防止把不同参数的结果混在一起：

```bash
python -u scripts/run_server_experiments.py --suite routing --phase both \
  --cases compute24 contact66 --updates 200 --output "results/routing_$(date +%Y%m%d_%H%M%S)"
python -u scripts/run_server_experiments.py --suite modules --phase both \
  --cases compute24 contact66 --updates 200 --output "results/modules_$(date +%Y%m%d_%H%M%S)"
python -u scripts/run_server_experiments.py --suite resource --phase both \
  --cases compute24 contact66 --updates 200 --output "results/resource_$(date +%Y%m%d_%H%M%S)"
```

`routing` 对比完整方法、snapshot_route、no_future；`modules` 对比完整方法、no_booking、no_node_mask；`resource` 对比 KKT 平方根与等分，按各自训练时资源配置评估。模块定义见 [METHOD.md](METHOD.md)。

## 5. 跨规模泛化

```bash
python -u scripts/run_server_experiments.py --suite scale --phase both \
  --cases scale48 --updates 200 --output "results/scale_$(date +%Y%m%d_%H%M%S)"
```

48 星训练，冻结权重和特征尺度后评估 24/48/72/96 星。训练与验证仍为 2026/100；此入口单独使用一个留出 seed 301，以符合冻结泛化接口的独立评估要求，不会启动多 seed 训练。结果相对同一留出 seed 的 48 星锚点报告代价变化，不用训练阶段 reward 直接计算泛化差距。

## 6. 负载与物理参数敏感性

必须先完成主实验，以下 `$RUN` 应指向保存了 compute24/contact66 的完整方法与 MLP `best.pt` 的根目录：

```bash
python -u scripts/run_server_experiments.py --suite sensitivity --phase evaluate \
  --cases compute24 contact66 --output "$RUN"
```

冻结模型，扫描负载、CPU、带宽、deadline 与前视窗口；参数点及结果保存在该根目录。与开发比较一致使用 seed 100，应标注验证复用，不叫独立测试。不要根据单个有利参数点删掉其他结果。

## 7. 场景审计

```bash
python -u scripts/run_server_experiments.py --suite audit \
  --cases compute24 contact66 --output "results/audit_$(date +%Y%m%d_%H%M%S)"
```

输出真实源分布、热点/全局计算负载、期限下界、拓扑变化与逐目标路径接触暴露。审计使用空系统与均匀抽样时刻，不把这些预测当成实际策略失败率。

任何批量命令加 `--dry-run` 可先打印实际子命令。新目录可避免覆盖历史服务器日志。

## 输出位置

```text
RUN/
├── train/<case>/<variant>/init_2026/
│   ├── best.pt / last.pt / update_*.pt
│   ├── episodes.csv / updates.csv / validation.csv
│   ├── training_manifest.json / training_summary.json / resolved_config.yaml
│   ├── learning_curve.png / routing_curve.png
│   └── topology.npz
├── eval/<suite>/<case>/init_2026/
│   ├── evaluation_study.json / seed_metrics.csv / aggregate.csv
│   ├── variant_mapping.json / paired_differences.csv
│   └── seed_100/<policy>/{tasks.csv,slots.csv,metrics.json}
├── logs/                  每个子进程的完整控制台文本
└── runner_status/         完成任务与输入指纹；相同任务可复用
```

总进程和直接执行的训练/评估另存控制台文本，启动时打印绝对路径，通常在输出根目录旁的 `console_logs/`。`tasks.csv` 区分 `requested_compute_sat`、`compute_sat`、`routing_rejection_reason` 与实际失败原因。单 seed 汇总不输出置信区间；新版结果与旧联合动作结果不要混写。

线程参数可不设置。默认 Torch CPU 线程为 1；需要比较更高设置可直接训练时 `--set rl.torch_threads=4`。OpenMP/MKL 线程和 Torch 线程不是 GPU 算力开关，优先用实际 FPS 比较，而非按 CPU 核数盲目增加。
