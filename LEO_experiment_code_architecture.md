# 实验工程结构

当前实现是“计算节点 PPO → 选定节点图路由 → KKT 资源分配”，与 [研究方案](LEO_research_system_model_and_experiments.md) 一致。

| 目录/文件 | 职责 |
|---|---|
| `leo_compute_routing/configs/` | 当前物理场景与学习配置继承 |
| `src/leo_routing/topology/` | 圆轨道 Walker、物理连边与缓存 |
| `src/leo_routing/tasks/` | 不可变整任务、固定随机流回放 |
| `src/leo_routing/network/contact_plan.py` | 有限未来接触与容量积分 |
| `src/leo_routing/models/destination_scorer.py` | 全部卫星共享评分、变长输出 |
| `src/leo_routing/agents/ppo_agent.py` | 请求节点采样、后置路由、冻结概率与 PPO |
| `src/leo_routing/routing/contact_aware_router.py` | 对一个已选节点进行图搜索及回退 |
| `src/leo_routing/routing/reservations.py` | 当前活动与批次前缀的预测预约 |
| `src/leo_routing/resource/` | 静态代理 KKT 闭式解、事件时重新共享 |
| `src/leo_routing/env/` | 时隙内事件推进、跨时隙工作量、指标和代价 |
| `src/leo_routing/evaluation/` | 一致回放、审计、冻结比较与泛化 |
| `scripts/run_server_experiments.py` | 顺序服务器调度、日志、恢复和实验命令 |

这里的 `src/` 路径均相对于 `leo_compute_routing/`。接口与约定见 [MODEL_AND_EXTENSION.md](leo_compute_routing/docs/MODEL_AND_EXTENSION.md)。旧联合动作模块已删除，不保留兼容分支；历史结果目录保留，schema=3 要求重新训练。

配置与代码严格使用 SI 单位。策略不修改物理拓扑或实际分配器。提交执行器的是解析后的 RoutingAction，强化学习 buffer 保存的是请求计算卫星，二者不能混淆。真实失败和路由拒绝分别保存，并使用统一代价。

环境无需新增第三方包，使用现有 PyTorch/NumPy/NetworkX/YAML。运行和验证见 [工程 README](leo_compute_routing/README.md) 与 [服务器命令](leo_compute_routing/docs/SERVER_EXPERIMENTS.md)。
