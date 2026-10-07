# 多 LEO 卫星边缘计算实验工程

当前采用 **GAT-PPO 选择计算卫星 → 预测接触感知图路由 → KKT 链路/CPU 分配**。强化学习只输出节点，路由在选择后执行；旧联合计算节点—路径方法及其配置已删除，旧结果保留。feature schema=3，旧权重需要重新训练。

- [三层方法与消融定义](docs/METHOD.md)
- [系统模型中文版](docs/SYSTEM_MODEL_ZH.md)、[英文版](docs/SYSTEM_MODEL.md)
- [中英文系统图与矢量文件](paper_figures/README.md)
- [服务器完整命令](docs/SERVER_EXPERIMENTS.md)、[环境依赖](docs/ENVIRONMENT.md)
- [场景与物理参数](docs/SCENARIOS.md)、[实验指标](docs/EXPERIMENTS.md)
- [当前验证记录](docs/VALIDATION.md)、[历史结果](docs/HISTORICAL_RESULTS.md)

训练固定初始化及任务 seed 2026，验证与开发评估使用同一个 seed 100。控制台显示进度、FPS、主要任务指标、PPO 数值和实际 GPU 使用，并保存完整文本。单 seed 不输出置信区间，验证复用不称为独立测试。

## 运行

在本工程目录及已安装 PyTorch 的 Python 环境执行：

```bash
python -m pytest -q
python -u scripts/train_ppo.py --config configs/rl_smoke.yaml --output results/new_smoke
python -u scripts/run_server_experiments.py --suite main --phase both \
  --cases compute24 contact66 --updates 40 --output results/new_hierarchical
```

每次使用新的输出目录；完整方法、MLP 对照及启发式采用共同任务/CPU/拓扑。40 updates 用于预实验，是否收敛以曲线和冻结比较为准。环境可继续使用现有 `.conda-env`，新架构不新增依赖。

## 职责

```text
src/leo_routing/
├── topology/    Walker 轨道、外生物理 ISL、缓存
├── tasks/       整任务生成与固定回放
├── network/     链路与有限未来 ContactPlan
├── compute/     当前 CPU / 在途剩余工作量
├── models/      GAT/MLP、共享节点评分、critic
├── agents/      节点 PPO、冻结特征、GAE、训练与 checkpoint
├── routing/     选定节点的图路由、预测预约、实际执行指令
├── resource/    KKT 平方根与等分资源共享
├── env/         连续事件、跨时隙服务、reward 与指标
├── baselines/   本地、节点启发式与相同路由的 node_greedy
├── evaluation/  公平回放、审计、消融、泛化与敏感性
└── utils/       输出、控制台与进度
```

任务完整输入逐跳转发，ISL 两个方向共享容量；数据到达后才计算。CPU 为处理器共享，Q/F 只作为特征/估计，不能叠加到实际完成时延。路由预测预约不锁定实际资源，仍可能因真实竞争失败或超期。KKT 最优性限于固定活动集合的静态代理。

`compute24` 是计算密集的 24 星场景，`contact66` 是高倾角、较低带宽的 66 星场景，两者使用同一方法。论文需要用 node_greedy 对照检验长期学习价值；短功能检查不证明方法已优于启发式。
