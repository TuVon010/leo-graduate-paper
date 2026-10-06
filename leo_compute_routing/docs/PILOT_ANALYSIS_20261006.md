# 20 updates 服务器结果复核与下一轮实验

复核日期：2026-10-06。数据来源：`results/server_pilot_20261005_182146/`。四次训练均为初始化 2026、20 updates、每 update 两个完整 episodes；五个验证 seed 为 100–104，测试仅 seed 201。服务器实际使用 NVIDIA GeForce RTX 4070 Ti / CUDA，不是 GPU 未启用造成的效果问题。

## 当前结论

**不能认定收敛；目前 GAT 有局部优势，但没有超过最强批次贪心。** 只有第 10、20 次两个验证点，无法判断稳定平台。训练日志有改善也不等于确定性测试策略收敛。

| 场景 / 方法 | 平均完成时延 s | P95 s | 按期成功率 | 每接纳任务成本 s | 平均跳数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| compute24 / GAT-PPO | 2.1334 | 4.1941 | 94.5954% | 2.2415 | 2.0357 |
| compute24 / MLP-PPO | 2.2208 | 3.7572 | 97.4752% | 2.2713 | 3.1138 |
| compute24 / computing_aware | 1.5379 | 2.8480 | 97.7895% | 1.5821 | 0.5491 |
| compute24 / batch_greedy | 1.3058 | 2.2206 | 99.5234% | 1.3153 | 0.5356 |
| contact66 / GAT-PPO | 1.1351 | 2.6242 | 98.8090% | 1.1590 | 0.3945 |
| contact66 / MLP-PPO | 1.6121 | 3.0016 | 98.8128% | 1.6358 | 1.4192 |
| contact66 / computing_aware | 1.1388 | 2.7644 | 98.3793% | 1.1710 | 0.1795 |
| contact66 / batch_greedy | 1.0615 | 2.4718 | 99.0050% | 1.0812 | 0.1901 |

表格使用验证选出的 best.pt：compute24 两个模型均为第 10 次；contact66 GAT 为第 20 次，MLP 为第 10 次。没有按测试结果重新选检查点。完整九方法表在原始 `eval/main/<case>/init_2026/seed_201/summary.csv`。

- compute24：GAT 比 MLP 平均时延低约 3.94%，但成功率低 2.88 个百分点、P95 更高，不能称为全面更好。相对 batch_greedy，时延高约 63.38%，成本高约 70.41%。
- contact66：GAT 比 MLP 平均时延低约 29.58%、成本低约 29.15%，成功率基本相同；相对 batch_greedy，时延高约 6.94%、成本高约 7.19%。GAT、MLP、future 贪心的实际路由失败均为零。
- local、shortest_offload 已经显著较差，无需进一步人为恶化。存在截尾时，完成任务平均时延不能独立评价所有接纳任务，应同时看每任务成本、成功率和截尾率。
- 两个场景九种方法的任务轨迹、CPU 容量、拓扑哈希分别一致，未发现不同方法使用不同物理输入。

## 收敛证据

验证均值如下。成本为原始 episode reward 按固定 normalizer 和接纳任务数换算，越低越好。

| 场景 / 编码 | 第 10 次成本 → 第 20 次成本 | 第 10 次成功率 → 第 20 次成功率 | 判断 |
| --- | --- | --- | --- |
| compute24 / GAT | 2.0560 → 2.3885 | 96.1004% → 94.0318% | 退步，未收敛 |
| compute24 / MLP | 2.0776 → 3.9859 | 98.5314% → 79.7074% | 明显退步 |
| contact66 / GAT | 1.6929 → 1.0668 | 98.7988% → 99.3438% | 正在改善，尚无平台证据 |
| contact66 / MLP | 1.5637 → 1.8254 | 99.1593% → 97.3190% | 退步，未收敛 |

![训练、验证和单 seed 测试诊断](../results/pilot_analysis_20261006/pilot_diagnosis.png)

浅线为每 update 训练 episode 的成本均值，实线为明确标注的 5-update 滑动均值，虚线仅连接实际存在的两个验证点。训练采样动作与验证 argmax 动作不同，曲线数值不能直接当成泛化差距。滑动均值也不能掩盖原始变化。

## 代码与检查点诊断

价值学习是当前需要优先修正的实现问题：

1. 原代码直接在原 reward 单位上最小化价值 MSE，和已标准化优势的策略损失共用图编码、上下文编码及全局梯度裁剪。价值损失数量级约数百至千，policy loss 约千分位。
2. 四组训练最后五次更新的 explained variance 都在零附近（绝对值约 $10^{-9}$–$10^{-8}$），说明价值预测几乎没有解释回报变化。compute24/GAT 的平均裁剪前梯度范数约 206，而最大范数设为 0.5。
3. 在训练 seed 0 的额外短前缀探针中，compute24 第 10 次检查点的价值预测约为 -118.0384，不同状态的标准差约 $8\times10^{-6}$，上下文 Tanh 饱和比例为 100%。该探针是训练集数值诊断，不是新的测试性能。
4. compute24/GAT 相比强基线产生更多多跳流量，CPU 利用率却基本相同。增加通信代价的路径偏好没有换来相应计算收益。

这些证据支持“共享表示饱和和价值目标尺度失衡妨碍学习”的诊断，但不能把所有失败都归因于一个因素。还存在长时序信用分配、联合批次动作方差、近均匀候选探索和预测服务误差。

可复现分析：

```bash
python scripts/analyze_pilot.py results/server_pilot_20261005_182146 --output results/pilot_analysis_recheck
python scripts/probe_pilot_policies.py results/server_pilot_20261005_182146 --slots 80 --seed 0 --output results/policy_probe_recheck.json
```

JSON 保存完整验证均值、同训练 seed 两次出现的差值、最后五次 PPO 指标和输入一致性检查。结果图和审计位于 `results/pilot_analysis_20261006/`。

## 已实现的优化

原配置和原检查点的行为保持兼容，新增配置单独训练，不覆盖原始数据。

| 改动 | 新值 | 作用及边界 |
| --- | ---: | --- |
| `rl.value_scale` | 100 | $V=s_v\tilde V$，价值损失为 $\mathrm{MSE}(V/s_v,G/s_v)$；GAE、reward 和评估仍用原单位。降低价值项对共享表示的数值压力，不改变物理目标 |
| `rl.completion_prior_strength` | 2 | 候选 logit 为 $z_\theta(c)-\alpha\log(1+\widehat T^{cal}_c/t_{scale})$。按已知接触/预约完成时间提供先验，神经网络学习可覆盖先验的残差 |
| learning rate | $10^{-4}$ | 降低单次参数变化，需通过验证而非测试调参 |
| entropy coefficient | 0.002 | 降低近均匀探索的激励；不是屏蔽合法较差动作 |
| minibatch slots | 64 | 相同物理时隙批次 PPO，提高梯度估计稳定性，不改变联合动作概率定义 |
| validation interval | 5 updates | 在 40 次预实验中获得 8 个验证点，便于观察退步与平台 |

新增 `contact_greedy` 无学习对照：使用相同 contact 候选、Shield、预约日历，直接选择预计完成时间最小的候选。若新策略只达到它的水平，收益应归于物理先验，不能称为 PPO 学出了额外优势。MLP 同样使用价值尺度和完成时间先验。

新配置为 `configs/experiments/contact_compute_tuned.yaml`、`contact_dynamic_tuned.yaml`；物理参数分别与原 compute24/contact66 相同。旧模型不包含先验，结果不能混称为同一版纯 GAT-PPO。控制台、manifest 和 checkpoint metadata 都记录先验强度和价值尺度，新增 `[CRITIC]` 输出原尺度 MSE、归一化 MSE、价值/目标标准差及梯度裁剪缩放。

## 是否调整场景

先在原场景检验学习修复，避免同时改算法与物理参数导致原因不可解释。contact66 原 seed 201 的全局提供负载约 0.394，batch_greedy 成功率 99.0%，确实接近饱和。

另提供一个预先指定的负载敏感性点 `contact_dynamic_balanced_tuned.yaml`：到达率由 11 增至 18 tasks/slot，其余轨道、任务大小、CPU、链路、deadline 不变，固定 reward normalizer 同比例改为 18。其目的为把全局提供负载提高到约 0.6–0.7，而不是改变某一基线的实现。热点本地过载必须明确报告。

已在独立诊断 seed 50 完成物理审计：全局期望提供负载 0.6054，最大热点本地负载 2.0417，乐观 CPU 下界判断的先天不可行任务比例为 0；预测窗口存在当前边移除的时间窗口比例为 13.33%。该场景全局算力尚有余量、热点需要卸载，但仅有 CPU 下界可行不等于所有任务都能按期完成。审计文件为 `results/pilot_analysis_20261006/balanced_calibration_seed50.json`。

此点应对所有方法共同应用，不得只保留排名有利的点。它不是实测负载，也不能保证 RL 胜出。保留原低负载和计算密集结果作为对照。

## 服务器命令

后续按用户要求改为每阶段仅一个固定 seed：模型初始化 2026、训练任务/CPU seed 2026、验证和当前开发评估均用 seed 100。三个 tuned 配置均采用 `train_seeds: [2026]`、`validation_seeds: [100]`（balanced 从 dynamic 继承）。不再轮换训练 seed 0–19 或用五个验证 seed。初始化只在模型创建时设置，PPO 动作采样随机状态在训练中正常推进，不在每个 update 重置。

每 update 仍采集两个完整 episodes，但都使用训练任务 seed 2026；40 updates 对应 80 个训练 episodes，每 5 updates 用同一个验证 seed 100 检查一次，共 8 个验证 episodes。验证用于选 best.pt；训练后与基线对比仍复用 seed 100，因此是开发评估，不是独立测试。当前先用这一个评估 seed 调试效果。

批量入口对全部 `_tuned` case 默认使用 seed 100，并将 `--allow-validation-reuse` 显式传入评估子命令。直接调用 `evaluate_ppo.py` 时，使用 `--seeds 100 --allow-validation-reuse`。日志和 `evaluation_study.json` 标记 `evaluation_role=validation_reuse`、`independent_test=false`；训练 seed 重叠仍会拒绝，避免训练数据被标为评估数据。未来需要独立测试时可显式指定新 seed，如 301。

先上传本次修改的 src、scripts、configs、tests、docs，无需重新安装环境。在服务器工程根目录的现有 conda 环境执行：

```bash
python -m pytest -q
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
STAMP=$(date +%Y%m%d_%H%M%S)
PILOT="results/tuned_pilot_${STAMP}"
RUN="results/tuned_full_${STAMP}"

# 第一轮只训练和验证；初始化/训练 seed=2026，验证 seed=100。
python -u scripts/run_server_experiments.py --suite main --phase train --cases compute24_tuned contact66_tuned --updates 40 --output "$PILOT"

# 验证曲线可接受后，冻结算法/物理设置，在新目录续训至总计 200 updates。
python -u scripts/run_server_experiments.py --suite main --phase train --cases compute24_tuned contact66_tuned --updates 200 --resume-root "$PILOT" --output "$RUN"

# 当前开发评估与验证统一 seed=100，明确标记为复用验证数据。
python -u scripts/run_server_experiments.py --suite main --phase evaluate --cases compute24_tuned contact66_tuned --updates 200 --test-seeds 100 --allow-validation-reuse --output "$RUN"
```

新增 tuned case 自动包含八种基线，包括 contact_greedy；原 case 仍默认七种。历史配置保留旧 seed 日程用于复现下载结果，后续使用上面的 tuned case。改变训练/验证 seed 日程同样要求从头训练；不要恢复此次 seed 修改之前的 tuned 检查点。

**不要从原 20-updates 的 last.pt 续训新配置。** 学习损失及策略定义改变，必须重新初始化。上述 resume 仅在新 tuned 配置内部使用，且仅改变目标 updates 和设备。

可选高负载场景先审计，再训练和开发评估：

```bash
python -u scripts/run_server_experiments.py --suite audit --cases contact66_balanced_tuned --output "results/balanced_audit_${STAMP}"
python -u scripts/run_server_experiments.py --suite main --phase train --cases contact66_balanced_tuned --updates 40 --output "results/balanced_pilot_${STAMP}"
```

拆分诊断 value scaling 和完成时间先验的消融，两组与 full 保持相同预算：

```bash
python -u scripts/train_ppo.py --config configs/experiments/contact_compute_tuned.yaml --set rl.completion_prior_strength=0 --set rl.updates=40 --output "results/no_prior_${STAMP}"
python -u scripts/train_ppo.py --config configs/experiments/contact_compute_tuned.yaml --set rl.value_scale=1 --set rl.updates=40 --output "results/raw_value_${STAMP}"
```

当前 seed 100 的对比只能说明复用验证任务上的表现；论文若需要独立测试结论，再使用未参与调参的 seed。当前开发阶段不增加测试 seed 数量。

## 验证范围

代码回归已通过 **112 项测试**，覆盖价值尺度与原单位对应、检查点往返、旧配置兼容、先验下节点/候选重编号等变、零残差与 contact_greedy 一致、屏蔽和批次预约不改变真实资源。

本地额外短轨道对照通过 `scripts/run_learning_diagnostic.py` 比较 raw / scaled / scaled_prior，采用相同初始化、训练 seed 日程和短预算，只使用验证 seed 100。此检查用于观察数值改善和机制行为，不替代完整 600 s、40–200 updates 训练，也不能证明正式测试优势或收敛。

该检查已实际完成：每种设置各训练 3 updates、每次 1 episode，到达窗口 32 s，最长排空 40 s，训练实际使用 seed 0、1、2；验证 seed 100 共 497 个任务，检查点按验证 reward 选择。

该记录使用 seed 修改前的配置，保持原始结果不变；现在重新运行诊断脚本会采用单一训练 seed 2026，不应期望逐项复现上面的历史数据。

| 短诊断设置 | 平均时延 s | 成功率 | 每任务成本 s |
| --- | ---: | ---: | ---: |
| raw（新小学习率、原价值尺度、无先验） | 6.3042 | 27.97% | 7.7449 |
| scaled（仅加价值尺度） | 5.9596 | 38.23% | 7.1950 |
| scaled_prior（价值尺度＋先验） | 1.1508 | 100.00% | 1.1508 |
| contact_greedy（先验，无学习） | 1.1516 | 100.00% | 1.1516 |
| batch_greedy_future | 1.1173 | 99.80% | 1.1213 |
| batch_greedy | 1.1160 | 99.60% | 1.1223 |

raw 三次更新的平均裁剪前梯度范数约 172.59，scaled 约 1.02；平均裁剪缩放由约 0.00292 提高到 0.49464。这证明数值压力已经减轻；scaled 的 explained variance 仍很低（第 3 次约 0.00136），不能声称价值网络已经学好。

带先验模型与 contact_greedy 的时延几乎相同，相比 batch_greedy_future 时延和成本仍略高。**当前短预算收益主要来自先验，尚未证明 PPO 带来额外收益。** 保存所有结果，不能只报告 100% 成功率。短窗口恰好具有少量断链事件，不代表完整 600 s 场景的长期失败率。

完整数据与终端文本在 `results/learning_diagnostic_20261006/` 及 `results/console_logs/learning_diagnostic_*.log`。可重跑：

```bash
python scripts/run_learning_diagnostic.py --updates 3 --slots 128 --output results/learning_diagnostic_recheck
```
