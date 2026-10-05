# 多 LEO 计算感知路由实验工程

第一阶段可运行工程：动态拓扑、星上任务、多跳路由、跨时隙资源竞争、计算队列、未来可行性预测、基线对比、消融与参数扫描。

对应父目录中的两份研究方案。原文档保持不变；建模约定见 [可行性分析](docs/FEASIBILITY.md)。本版已实现 **候选 MLP-PPO、边特征 GAT-PPO、预测可行性 mask 和批次自回归决策**，含训练、验证选模、检查点恢复和独立测试入口。使用方法见 [RL.md](docs/RL.md)。

论文系统模型初稿提供 [英文版](docs/SYSTEM_MODEL.md) 与 [中文版](docs/SYSTEM_MODEL_ZH.md)，包含动态轨道拓扑、整任务卸载、跨时隙共享服务、预测可行性与优化问题，并附代码对应说明。

项目环境已安装 PyTorch 2.7.1+cu126，RTX 4060 CUDA 验证成功，**72 项测试通过**；已跑通 MLP CPU / GAT GPU 短训练与独立 seed 评估。此前五 seed 的 24 星启发式预实验也已保留。问题复核见 [项目复核](docs/PROJECT_REVIEW.md)，各阶段检查见 [验证记录](docs/VALIDATION.md)。默认轨道配置仍为低负载对照，RL 短训练检查不代表正式论文性能。

## 配置环境后先运行

完整环境说明在 [ENVIRONMENT.md](docs/ENVIRONMENT.md)。在工程目录执行：

```powershell
Set-Location E:\postgraduateLife\paper2\leo_compute_routing
conda create --prefix .conda-env python=3.11 pip -y
conda run --prefix .conda-env python -m pip install -r requirements-dev.txt
conda run --prefix .conda-env python -m pytest -q
conda run --prefix .conda-env python scripts/run_baselines.py --config configs/smoke.yaml --output results/smoke
```

`smoke.yaml` 是 6 节点合成网络的功能检查。真实轨道近似模型使用：

```powershell
python scripts/run_baselines.py --config configs/base.yaml --output results/walker24
```

较大的实验：

```powershell
python scripts/run_baselines.py --config configs/medium.yaml --output results/walker48
python scripts/run_baselines.py --config configs/large.yaml --output results/walker72
```

脚本自带包路径引导，以上命令不要求先安装本项目。需要在其他工程中导入时，可执行 `python -m pip install -e .`。

## 目录和职责

```text
leo_compute_routing/
├── configs/                    参数、场景配置及继承
├── src/leo_routing/
│   ├── config.py               配置合并、校验、实验指纹
│   ├── topology/               Walker 位置、物理连边、拓扑缓存
│   ├── tasks/                  不可变 Task、独立随机流、任务回放
│   ├── network/                链路公式、K 条简单路径、未来接触检查
│   ├── compute/                已到达 CPU 的剩余工作量及在途工作量
│   ├── resource/               链路/CPU 闭式分配与等分分配
│   ├── routing/                统一动作、候选构建、显式 fallback
│   ├── env/                    事件推进、观测、reward、统计
│   ├── baselines/              七种策略，包含批次虚拟预约
│   ├── models/                 MLP/GAT 编码、task/path 编码、候选评分、critic
│   ├── agents/                 预约特征、masked PPO、GAE、训练与检查点
│   ├── evaluation/             参数审计、配对多 seed 统计、回放、消融与扫描
│   └── utils/                  JSON/YAML/CSV 写出
├── scripts/                    命令行运行入口
├── tests/                      模型公式及系统边界条件验证
├── docs/                       分析、环境、模型约定和后续接口
├── requirements*.txt
└── pyproject.toml
```

`env/leo_env.py` 只协调流程。`env/event_engine.py` 管理真实服务进度，`resource/` 负责分配，策略只读 `Observation` 并返回动作。增加策略无需修改物理执行流程。

## 实现约定

- 全部使用 SI 单位：bit、CPU cycles、cycles/s、bit/s、m、s。deadline 是相对于任务产生时刻的持续时间。
- 简化 Walker-Delta 使用圆轨道二体运动和 ECI 坐标；检查地球遮挡、最大距离和节点度数。不会补虚构链路来强制连通。
- 每个时隙边界统一决策当前任务批次；时隙内按传输完成、传播完成、CPU 完成、deadline 等事件推进。
- 存储转发：每一跳完整发送输入数据。每条 ISL 两个方向共享一个容量预算；本版没有全双工独立容量、输出回传、任务拆分、迁移或断链重路由。
- 数据到达目标后进入 CPU；在途任务单独记录。既有任务和新任务一起占用资源，跨时隙不重置进度。
- CPU 为处理器共享，各活动任务持续获得分配；`Q/F` 仅用于启发式负载估计，**不会额外加到事件仿真的实际时延上**。
- 资源闭式解用于固定活动集合的静态时延子问题；成员变化时重新分配。它不是动态全局任务完成时间的最优解。
- 候选预测不是成功保证。真实共享速率可能比参考速率低；实际断链仍会产生失败记录。

详见 [模型约定和扩展接口](docs/MODEL_AND_EXTENSION.md)。

## 基线

| CLI 名称 | 决策规则 |
|---|---|
| `local` | 全部本地计算 |
| `shortest_offload` | 有远程候选时，选参考网络时延最小的远程路径；无远程路径时本地回退 |
| `least_load` | 选择已排队及已承诺在途工作量 / CPU 最小的目标，然后选择网络时延最小的路径 |
| `computing_aware` | 最小化网络时延 + 负载时间 + 独占 CPU 执行时间的估计 |
| `computing_aware_future` | 同上，并使用未来链路可行性和 deadline 预测筛选 |
| `batch_greedy` | 按 deadline 顺序选择，同批选择累计 CPU 和共享链路工作量预约 |
| `batch_greedy_future` | 同上，筛选未来可行性并考虑预约后的 deadline 代价 |

除 future 策略外，环境将 lookahead 置为 0，避免未来特征泄漏。环境不强制所有策略使用 mask，使消融成立。所有候选都不可行时，future 策略显式允许本地回退，但本地不会被标记为“预测可行”。

仅按通信时延比较本地和远程时，本地通信时延为零，会退化成 `local`。因此本版将第二条基线明确命名为 `shortest_offload`，需要在论文中说明其目标节点选择规则。

## 拓扑缓存、复现与输出

```powershell
python scripts/generate_topology.py --config configs/base.yaml --output data/topology/base.npz
python scripts/run_baselines.py --config configs/base.yaml --topology-cache data/topology/base.npz --output results/cached
```

缓存会检查拓扑参数、时隙长度及总覆盖长度的指纹。不匹配时直接报错，请换一个缓存路径重新生成。不要使用不同场景的缓存。

一次对比会保存：

```text
results/<experiment>/
├── resolved_config.yaml        最终参数
├── manifest.json               依赖版本、配置/任务/CPU 指纹、拓扑诊断
├── calibration.json            源概率、负载、deadline 下界、时间尺度与接触诊断
├── topology.npz                本次实际使用的拓扑缓存
├── task_trace.json             本次实际使用的固定任务序列
├── cpu_capacities.json         固定的异构 CPU 容量
├── summary.csv / summary.json  算法汇总
└── <algorithm>/
    ├── metrics.json
    ├── tasks.csv               逐任务状态、完成时延、失败原因
    └── slots.csv               队列、reward、完成数、失败数等
```

同一次对比的任务、CPU 与拓扑完全相同。可以通过保存的 `resolved_config.yaml`、`topology.npz` 和 `task_trace.json` 回放：

```powershell
python scripts/run_baselines.py --config results/cached/resolved_config.yaml --topology-cache results/cached/topology.npz --task-trace results/cached/task_trace.json --output results/replay
```

时延与任务结果应可复现；墙钟耗时不要求完全相同。配置 seed 改变会生成新的任务和 CPU 分布。

## 消融、扫描和画图

新增参数审计和多 seed 配对统计入口：

```powershell
python scripts/audit_scenario.py --config configs/base.yaml --output results/base_audit.json
python scripts/run_multiseed.py --config configs/base.yaml --seeds 42 43 44 45 46 --output results/multiseed_new
```

多 seed 输出 `aggregate.csv`、`paired_differences.csv` 并保留各 seed 回放材料。均值按 seed 等权；CI 重采样 seed，不重采样相关任务。默认参照为 `batch_greedy`，拒绝覆盖非空目录。旧过载参数使用 `configs/legacy_overload.yaml`。

```powershell
python scripts/run_ablation.py --config configs/contact_stress.yaml --output results/ablation
python scripts/run_sensitivity.py --config configs/base.yaml --parameter tasks.arrival_rate_per_slot --values 4 8 12 20 --seeds 42 43 44 --output results/load_sweep
python scripts/run_sensitivity.py --config configs/contact_stress.yaml --parameter routing.lookahead_slots --values 0 1 3 8 24 --seeds 42 43 44 --algorithms computing_aware_future --output results/horizon_sweep
python scripts/plot_results.py results/walker24/summary.csv
```

数组参数也能扫描，例如 PowerShell 中使用 `--values '[500000.0,2000000.0]' '[500000.0,5000000.0]'`。所有脚本支持重复 `--set section.key=value` 覆盖参数。

本版消融包含 future、等分资源、deadline mask；移除 future 时保留相同的 deadline mask 和 fallback，避免把两者效果混在一起。扫描 H=0 也保留 deadline 筛选，仅关闭未来接触检查。**没有将未实现的 GAT/PPO 消融伪装成实验结果**。`contact_stress.yaml` 是合成链路压力测试，不能替代真实轨道模型的论文证据。

## 指标怎么读

`mean_completion_delay_s` 只统计实际完成的任务；`success_rate` 要求任务完成且满足 deadline。失败与删失任务不会被记成零时延。比较平均时延时必须一起看成功率、完成率、路由失败率与 `censored_rate`。

`mean_sojourn_s` 统计完成/失败/丢弃/删失时的驻留时间；删失项只是截至仿真结束的观测长度。`deadline_violation_rate` 记录曾超过 deadline 的任务，可能和之后发生的路由失败重叠。任务统计采用固定窗口内到达的任务队列，排除 warmup；`offloaded_route_failure_rate` 以其中卸载任务为分母。

默认超过 deadline 后继续执行，以便观测迟到完成时间。停止产生新任务后会进入 drain 阶段；若仍未完成则标为 `censored`。出现删失时，应扩大 `simulation.drain_slots` 或明确报告。压力测试则开启 `drop_at_deadline`，到期释放资源。

链路和 CPU 利用率按固定窗口 `[warmup_slots × slot_seconds, slots × slot_seconds)` 的容量积分计算，排除 drain；窗口内的 warmup 任务仍参与资源竞争。队列指标也使用同一窗口。旧全 episode 利用率保留为 `episode_cpu_utilization`、`episode_link_utilization`。`policy_select_ms_per_task` 只计策略选择，候选/观测构造耗时单独记录，不能冒称完整在线推理时间。

## 后续阶段

RL 入口已实现，可先做短合成检查，再运行完整轨道训练：

```powershell
conda run --prefix .conda-env python scripts/train_ppo.py --config configs/rl_smoke.yaml --set rl.encoder=mlp --set rl.device=cpu --output results/mlp_smoke_new
conda run --prefix .conda-env python scripts/train_ppo.py --config configs/rl_smoke.yaml --set rl.device=cuda --output results/gat_smoke_new
conda run --prefix .conda-env python scripts/evaluate_ppo.py --checkpoints results/mlp_smoke_new/best.pt results/gat_smoke_new/best.pt --seeds 201 202 --output results/rl_test_new
```

完整训练配置为 `configs/mlp_ppo.yaml`、`configs/ppo.yaml`；依赖为 `requirements-rl.txt`，本机固定 CUDA 构建为 `requirements-rl-cu126.txt`。checkpoint 只按验证集 reward 选择，独立测试拒绝复用训练/验证 seed。训练参数、恢复和消融详见 [RL.md](docs/RL.md)。后续仍需校准场景、正式训练、Oracle 和跨拓扑泛化。
