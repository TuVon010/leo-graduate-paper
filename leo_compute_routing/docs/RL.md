# MLP-PPO 与 GAT-PPO

本版新增接触窗口候选、预测预约 Shield 与冻结跨规模评估，完整说明和最新命令见 [CONTACT_METHOD.md](CONTACT_METHOD.md)。推荐 `configs/contact_ppo.yaml`。特征 schema=2，旧 schema=1 检查点需要重新训练。

本阶段已实现任务条件候选策略、两种编码器、预测可行性 mask、同批预约特征、联合 PPO 更新、训练/验证、检查点恢复及独立测试入口。物理环境和资源执行规则沿用已有引擎，基线没有改动。

训练控制台现在显示 rollout/优化/验证阶段、FPS、ETA、主要实验指标、PPO 诊断和实际 CPU/GPU 状态；默认每 10 秒刷新阶段进度，可用 `--log-interval-seconds 5` 调整。train/evaluate/generalization 会自动保存 stdout、stderr 和异常文本到输出目录同级 console_logs；批量入口实时转发子日志并保存每次完整控制台。详见 [服务器命令与日志说明](SERVER_EXPERIMENTS.md)。

算法依据为 [PPO 原论文](https://arxiv.org/abs/1707.06347) 和 [GAT 原论文](https://arxiv.org/abs/1710.10903)。本项目使用带边特征的多头 GAT 变体和自回归批次动作，不声称与论文网络结构完全相同。

## 依赖和设备

工程原有 `.conda-env` 可继续使用。新增依赖仅为 PyTorch，不需要 PyTorch Geometric、Gymnasium、torchvision 或 torchaudio。GAT 直接使用 PyTorch 实现。

本机 RTX 4060 Laptop GPU（8 GB），驱动 576.02，选用 PyTorch 2.7.1 CUDA 12.6 官方构建。固定构建见 `requirements-rl-cu126.txt`，一般 RL 依赖见 `requirements-rl.txt`。安装方式参照 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/)。

2026-10-05 已安装成功，项目 Python 3.11.17 中 `torch.cuda.is_available()` 为 True，GPU 前向/反向及 NumPy 2.4.6 互操作通过。`pip check` 无依赖冲突，72 项测试通过。完整环境版本快照保存为 `requirements-rl-lock.txt`；其中 Torch 已用固定官方版本替换本地 wheel 绝对路径，便于在其他同类机器复现。

官方 wheel 已缓存于 `data/wheels/torch-2.7.1+cu126-cp311-cp311-win_amd64.whl`，并与官方响应中的 SHA256 校验一致。如在线下载再次停滞，可直接从本地安装：

```powershell
conda run --prefix .conda-env python -m pip install "data/wheels/torch-2.7.1+cu126-cp311-cp311-win_amd64.whl"
```

```powershell
Set-Location E:\postgraduateLife\paper2\leo_compute_routing
conda run --prefix .conda-env python -m pip install -r requirements-rl-cu126.txt
conda run --prefix .conda-env python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

其他无 GPU 机器可先安装官方 CPU wheel，再装通用 requirements：

```powershell
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-rl.txt
```

`rl.device=auto` 自动选 CUDA 或 CPU；也可指定 `cpu`、`cuda`、`cuda:0`。小图和大量逐任务调用有额外 GPU 调度开销，GPU 不一定比 CPU 快。默认 Torch CPU 线程数为 1，可由 `rl.torch_threads` 调整。

## 文件职责

| 文件 | 职责 |
|---|---|
| `agents/settings.py` | RL 参数校验、future/mask 环境开关 |
| `agents/features.py` | 固定量级标准化、不可变图/任务/候选快照、同批预约 |
| `models/gat_encoder.py` | 含 self-loop、边特征与残差的多头邻域注意力 |
| `models/candidate_scorer.py` | task/path MLP、源/目标/路径/全局表示融合评分 |
| `models/actor_critic.py` | MLP 或 GAT、图 pooling、masked categorical、critic |
| `agents/rollout_buffer.py` | 保存物理时隙 transition，GAE 与 return |
| `agents/ppo_agent.py` | 采样、联合动作重评分、clip 更新、KL/梯度检查、检查点 |
| `agents/trainer.py` | 完整 episode 收集、训练 seed 轮换、验证、日志与最佳检查点 |
| `evaluation/rl_evaluator.py` | 冻结检查点与既有基线在共同 trace 上做独立测试 |
| `scripts/train_ppo.py` | 训练 CLI |
| `scripts/evaluate_ppo.py` | 测试 CLI |
| `scripts/plot_training.py` | 原始训练/验证 reward 和成功率曲线 |

## 状态和网络

节点使用 8 维特征：已有 `[Q/F, F/mean(F), inflight/F, x/r, y/r, z/r]` 加本批源任务的计算工作量/F、任务数。边使用距离、容量、当前工作量/容量、预测窗口可用比例。

12 维全局摘要包括任务产生阶段进度、批次规模、CPU/传输/传播阶段任务数、已违约数、活动剩余工作、输入需求与 deadline。critic 用节点 mean/max pooling 与这些摘要。摘要没有完整保留所有活动任务的路径和残余状态，因此这是函数近似的状态表示，不是已证明的完整充分统计量。

task MLP 使用数据量、cycles/bit、deadline、独占执行量级与本批选择位置；不把 satellite ID 当连续学习特征。path MLP 融合 18 维候选特征和路径节点 mean pooling：8 个原候选特征、2 个接触余量、3 个原预约代理、5 个日历完成时间/期限/接触/容量/覆盖特征。

Actor 拼接任务、源、目标、路径和全局 embedding 后输出每个候选 logit。候选数、星数可以变化，网络没有固定候选输出维度。MLP 逐节点编码；GAT 默认 2 层、4 头、hidden=64，并读链路特征。两者共用后续评分头和 critic 结构。

时间以 checkpoint 中的 `time_scale_seconds` 标准化；数据量和复杂度使用训练配置的固定尺度。计数类全局量按当前星数与训练期固定逐星尺度归一化，部分长尾特征做 signed log1p。测试不拟合新的统计。GAT 使用 dense 邻域注意力，已验证同一权重在 24/48/72/96 星可执行，但其 O(N²) 成本不能忽略。

## 批次动作与 mask

物理时隙边界按 `(deadline, task_id)` 排序，逐任务构造分布并选择。选完更新目标 CPU cycles 和无向路径 bits 预约，供后续选择使用；整批结束后调用一次 `LeoEnv.step`。预约不推进时间，也不提前消耗 CPU 或链路资源。

预约是策略特征和 deadline 代价代理，不是精确共享完成时间。Actor 自己学习评分，没有将贪心排名或基线动作硬编码为输出。

`rl.shield_mode=mask` 保留原预测 mask 和粗工作量代价；`contact` 使用当前活动任务与批次前缀形成的时间日历重算接触/覆盖/期限；`none` 不屏蔽动作。`use_mask=false` 同样关闭屏蔽。全不可行时显式允许本地候选，记录 fallback；这不表示满足 deadline。future 关闭时 H=0，不读取未来拓扑。训练 update 使用采样时保存的候选和 mask，不重新访问实时环境。

一个时隙动作是整个自回归批次。联合概率为各条件概率的乘积，log probability 为其和。**PPO ratio 对联合动作计算一次，不能把同批任务视为独立物理 transition。** 空批次和只有强制动作的时隙仍训练 critic、传播 GAE，但不产生 actor loss。熵项采用已访问前缀上的条件熵均值作为正则估计。

## PPO 和终止语义

包含 GAE、advantage 标准化、clipped surrogate、value MSE、熵正则、梯度裁剪和联合 KL 提前停止。小批次由物理时隙组成，保留 ragged 图和候选。网络不使用 dropout，确保首次重评分与采样概率一致。

训练每个 update 收集完整 episodes，可跨不同 seed，但 GAE 不跨 episode 边界。空任务时隙和 drain reward 全部保留。当前 `truncated` 意味着达到 drain 上限、剩余任务已被 censor，是有限时域终端结果，bootstrap=0；不能把它误当继续运行的普通 rollout 时间截断。

奖励使用现有环境的 holding cost 与 deadline/断链惩罚，没有根据测试结果增加有利于 proposed 的奖励项。该目标与“只看已完成任务平均时延”并不完全相同，评估时仍需同时看成功率、失败率、删失率。

## 先跑功能检查

`rl_smoke` 为短合成网络检查，不是轨道论文结果。建议先 CPU，随后验证 CUDA。

```powershell
conda run --prefix .conda-env python -m pytest -q
conda run --prefix .conda-env python scripts/train_ppo.py --config configs/rl_smoke.yaml --set rl.device=cpu --output results/mlp_smoke_new --set rl.encoder=mlp
conda run --prefix .conda-env python scripts/train_ppo.py --config configs/rl_smoke.yaml --set rl.device=cuda --output results/gat_smoke_new
conda run --prefix .conda-env python scripts/evaluate_ppo.py --checkpoints results/mlp_smoke_new/best.pt results/gat_smoke_new/best.pt --seeds 201 202 --output results/rl_test_new
conda run --prefix .conda-env python scripts/plot_training.py results/gat_smoke_new
```

完整入口：

```powershell
conda run --prefix .conda-env python scripts/train_ppo.py --config configs/mlp_ppo.yaml --output results/mlp_train_new
conda run --prefix .conda-env python scripts/train_ppo.py --config configs/ppo.yaml --output results/gat_train_new
conda run --prefix .conda-env python scripts/evaluate_ppo.py --checkpoints results/mlp_train_new/best.pt results/gat_train_new/best.pt --seeds 201 202 203 204 205 --output results/rl_test_full_new
```

多训练初始化 seed 使用 `--set rl.seed=2027` 等建立独立输出目录。task/CPU 的训练 seed、验证 seed 与网络初始化 seed 是不同概念。默认 20 个训练场景 seed、5 个验证 seed；每个 update 2 episodes，轮换场景 seed。训练/验证拓扑目前共用同一物理配置，代码仅强制 seed 分离，跨拓扑泛化需要显式给评估脚本 `--config`。

## 检查点、恢复和独立测试

输出包括 `episodes.csv`、`updates.csv`、`validation.csv`、`training_manifest.json`、拓扑、`best.pt`、`last.pt` 和周期检查点。最佳模型只按验证集平均 episode reward 选择，测试集不选模型。

完整 episodes 更新结束后保存 optimizer、NumPy permutation RNG、Torch RNG、训练 episode 计数，便于恢复。不保存 NetworkX/环境对象，使用 `weights_only=True` 加载张量与基础容器。恢复允许更改设备和目标 updates，不能悄悄更改网络、尺度、场景或目标。

```powershell
conda run --prefix .conda-env python scripts/train_ppo.py --config configs/ppo.yaml --resume results/gat_train_new/last.pt --set rl.updates=300 --output results/gat_resume_new
```

测试入口拒绝与任一检查点声明的 train/validation seed 重叠。冻结权重，逐条件 argmax；每个测试 seed 的基线与模型共享任务、CPU、拓扑和执行器，并保存 checkpoint SHA256、原始任务/时隙 CSV、seed 均值、配对 CI。不同设备上的浮点结果可能不同，不承诺跨硬件逐比特相同。

## 方法消融

```powershell
python scripts/train_ppo.py --config configs/ppo.yaml --set rl.encoder=mlp --output results/ablate_mlp
python scripts/train_ppo.py --config configs/ppo.yaml --set rl.use_future=false --output results/ablate_future
python scripts/train_ppo.py --config configs/ppo.yaml --set rl.use_mask=false --output results/ablate_mask
python scripts/train_ppo.py --config configs/ppo.yaml --set rl.use_reservations=false --output results/ablate_reservations
```

每个变体应独立训练、使用相同预算和 seed 划分，再在相同 held-out 场景比较。功能检查通过只说明训练链路成立，不代表已收敛或优于基线；当前短 smoke 检查点不应用于论文结论。

本机实际验证结果保留在 `results/rl_verification_20261005/`：`mlp_project` 是 CPU 的 2-update 训练，`gat_project` 是 CUDA 的 2-update 训练，`evaluation_project` 对两种冻结检查点与原有五种基线做 seed=201、202 的共同 trace 评估。训练 seed 为 42、43（配置另声明 44），验证 seed 为 101。`gat_project/learning_curve.png` 已生成并检查。两次训练 episode 使用不同 seed，不能将短曲线的变化全部归因为学习收益。
