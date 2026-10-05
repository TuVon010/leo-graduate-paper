# 模型约定与后续算法接口

## 时序

在时隙 `t` 边界：先结算此前区间的事件，再观察 `G(t)`、现存任务、CPU 剩余工作量，并读取该边界新到达的任务。

每个策略一次返回本批次所有任务的 `(计算目标, 完整路径)`。环境验证整个批次后统一加入执行器，再推进 `Δt`。CPU/ISL 资源只分配给当时真正处于对应服务阶段的任务。

状态机：

```text
本地：cpu → completed
远程：tx → prop → tx → ... → prop → cpu → completed
tx 且当前链路丢失：route_failed
到 deadline：记录 violation；若开启丢弃则 timed_out
达到 drain 上限：censored
```

传播延时取发送完成前最后服务区间对应快照的距离。传输结束恰好在快照边界时，先完成发送再处理新快照断链。没有对已失效路径重新路由。

每个事件区间内，资源速率是常数。拓扑切换、任务进入/离开 link 或 CPU 时重新求解。任务工作量大小用于平方根比例，已完成的部分仅从残余量中扣除；这与固定集合静态子问题一致，但不是动态最优控制。

数值比较使用秒级 `1e-10` 容差；对于大绝对时间上低于浮点可分辨时间的微小残余服务，执行器结算剩余量并在同一可表示时刻完成，避免零进度循环。资源容量检查容差为相对 `1e-12`。

## 候选路径与预测

默认 `path_backend=bounded`：在最大跳数内一次遍历源节点的简单路径，按 `reference_data_bits/R + distance/c` 排序，对每个候选目标取 K 条。对于最大度数 4、最大跳数 4，遍历树很小，避免每个目标重复跑 Yen。达到 `path_expansion_limit` 时明确报错，不悄悄截断。

可设 `path_backend=yen` 使用 NetworkX `shortest_simple_paths`，最多检查 `path_search_limit` 条，再筛选跳数；因此可能少于 K 条。默认 bounded 方法得到的是跳数限制内的精确 K 条简单路径，但权重使用参考任务大小，未声称是每种任务数据量下全部路由的精确最优集合。

同一时隙同一源节点复用路径列表。不同任务仍有独立的通信预测、CPU 工作量估计和 deadline margin。

参考传输速率为最大容量的 `reference_rate_fraction` 倍。未来检查考虑所覆盖快照内的最小容量，检查整个发送区间；不会读取未来新任务。`lookahead_slots=H` 时，最多读当前及后 H 个快照；H=0 禁用未来检查。

CPU 工作量估计采用 `(当前 CPU 剩余 cycles + 已承诺在途 cycles) / F`。它是拥塞代理，不是服务纪律下的严格等待时间。RL 与批次贪心另在策略层逐次加入已选择任务的 CPU/链路预约；原独立决策启发式仍只使用初始快照。

## 统一接口

```python
from leo_routing.config import load_config
from leo_routing.env.leo_env import LeoEnv
from leo_routing.baselines import make_policy

config = load_config("configs/smoke.yaml")
env = LeoEnv(config)
policy = make_policy("computing_aware_future")
obs, info = env.reset()
while True:
    actions = policy.select(obs)   # {task_id: candidate_index}
    obs, reward, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        break
print(info["episode_metrics"])
```

可以传入候选中的 `RoutingAction` 替代索引。环境检查 task ID、当前批次和候选一致性，但不会替策略强制启用预测 mask。

`LeoEnv` 提供 reset/step 五元组协议，**不是 Gymnasium 子类**。后续如使用 Gymnasium，应新增明确的可变图/序列空间适配层；不能仅因为返回元组类似，就把它传给要求 `gymnasium.Env` 的训练框架。

## Observation

| 字段 | 内容 |
|---|---|
| `slot`, `time_seconds` | 轨道快照位置与连续时间 |
| `graph` | 当前 NetworkX 图的只读结构副本 |
| `tasks` | 当前任务批次，Task 不可变 |
| `candidates[task_id]` | 可变数量 CandidateAction，索引 0 始终本地 |
| `feasibility_masks[task_id]` | 有效采样 mask，含全不可行时的本地 fallback |
| `fallback_task_ids` | 明确列出发生 fallback 的任务 |
| `cpu_capacities`, `cpu_queue_cycles`, `inflight_cycles` | CPU 容量、已送达剩余量、尚在网络的已承诺量 |
| `node_features` | `[Q/F, F/mean(F), inflight/F, x/r, y/r, z/r]`，shape `[S,6]` |
| `edge_index` | 双向展开的图边索引，shape `[2,2E]` |
| `edge_features` | `[归一化距离, 归一化容量, 活动剩余 bits/R, 窗口可用比例]` |
| `task_features` | `[D/Dmax, C/Cmax, deadline_s, 归一化 source_id]` |
| `candidate_features[task_id]` | `[归一化 hops, route_s, workload_s, execution_s, margin_s, 归一化 bottleneck, fully_checked, topology_feasible]` |
| `active_jobs` | 阶段、路径、hop、剩余 bits/cycles、原始工作量、传播剩余时间、deadline 剩余时间与报告状态 |

时间特征仍以秒表示。后续训练可以加入固定量级标准化或保存运行统计，不应对每个算法使用不同尺度。empty batch 的任务特征 shape 为 `[0,4]`；无链路时边特征 shape 为 `[0,4]`。

## PPO/GAT 实现约定

1. 一个真实时隙是一个 transition。无新任务的时隙仍可能有服务、失败与 reward，应保留它的时间意义。
2. Actor 按 task-conditioned candidates 打分，同一时隙按 deadline 顺序自回归采样，将此前选择作为预约特征。整批联合 log probability 为条件 log probability 的和；PPO ratio 对整个物理时隙计算。Critic 使用图 pooling、批次任务与活动任务阶段摘要。
3. 保存采样时完整候选特征、有效 mask、动作索引、旧 log probability 与 value；更新时不要用已经变化的拓扑重建 mask。
4. 训练收集完整 episodes，GAE 不跨 episode。当前 drain 截断实际 censor 剩余任务，是有限时域终端，bootstrap=0；普通仍继续运行的 rollout 截断则需要 bootstrap，两者不混淆。
5. 用环境共同的分配器执行动作；不得只给 proposed 算法使用更有利的队列或资源规则。
6. MLP 与 GAT 共用评分/值函数接口，future、mask、预约分别可关闭。正式消融需要独立训练并报告完整候选构造成本。

`models/` 和 `agents/` 现已实现；`EventEngine` 和 `LeoEnv` 继续不依赖 PyTorch。训练、验证、恢复及独立测试的入口与限制见 [RL.md](RL.md)。
