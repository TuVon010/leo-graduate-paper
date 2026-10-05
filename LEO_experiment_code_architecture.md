# 多LEO计算感知路由实验工程代码结构设计

> **用途**：第二研究点实验工程设计说明  
> **对应研究主题**：多LEO协同边缘计算中的计算感知路由与资源协同优化  
> **目标算法**：Task-aware GAT-PPO + Predictive Feasibility Mask + Structured Resource Optimization

---

# 1. 工程总体目标

整个实验工程需要实现如下数据流：

```text
卫星拓扑生成
    ↓
任务生成
    ↓
候选计算卫星
    ↓
K-shortest候选路径
    ↓
未来拓扑可行性筛选
    ↓
GAT-PPO选择(计算卫星, 路径)
    ↓
链路资源分配
    ↓
CPU资源分配
    ↓
计算任务真实完成时延
    ↓
计算队列更新
    ↓
Reward
    ↓
下一时隙
```

工程设计原则：

1. **系统模型与强化学习算法分离**；
2. **拓扑、任务、资源、路由、环境、Agent模块解耦**；
3. **所有baseline共用同一个环境**；
4. **所有实验参数集中放在配置文件**；
5. **先做环境，再做baseline，最后做GAT-PPO**；
6. **不要一开始就把所有代码写进一个env.py**。

---

# 2. 推荐目录结构

```text
leo_compute_routing/
│
├── README.md
├── requirements.txt
│
├── configs/
│   ├── base.yaml
│   ├── small.yaml
│   ├── medium.yaml
│   ├── large.yaml
│   └── ppo.yaml
│
├── data/
│   ├── topology/
│   ├── tasks/
│   └── cache/
│
├── src/
│   │
│   ├── topology/
│   │   ├── walker.py
│   │   ├── orbit_generator.py
│   │   ├── graph_builder.py
│   │   └── topology_cache.py
│   │
│   ├── network/
│   │   ├── link_model.py
│   │   ├── path_model.py
│   │   ├── ksp.py
│   │   └── feasibility.py
│   │
│   ├── tasks/
│   │   ├── task.py
│   │   ├── task_generator.py
│   │   └── task_queue.py
│   │
│   ├── compute/
│   │   ├── satellite_compute.py
│   │   └── compute_queue.py
│   │
│   ├── resource/
│   │   ├── link_allocator.py
│   │   ├── cpu_allocator.py
│   │   └── joint_allocator.py
│   │
│   ├── routing/
│   │   ├── candidate_builder.py
│   │   ├── action_builder.py
│   │   └── route_cost.py
│   │
│   ├── env/
│   │   ├── leo_env.py
│   │   ├── state_builder.py
│   │   ├── reward.py
│   │   └── metrics.py
│   │
│   ├── models/
│   │   ├── gat_encoder.py
│   │   ├── task_encoder.py
│   │   ├── path_encoder.py
│   │   ├── candidate_scorer.py
│   │   └── value_network.py
│   │
│   ├── agents/
│   │   ├── ppo_agent.py
│   │   ├── rollout_buffer.py
│   │   └── trainer.py
│   │
│   ├── baselines/
│   │   ├── local_only.py
│   │   ├── shortest_path.py
│   │   ├── least_load.py
│   │   ├── computing_aware.py
│   │   └── vanilla_ppo.py
│   │
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── ablation.py
│   │   └── sensitivity.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── logger.py
│       ├── io.py
│       └── math_utils.py
│
├── scripts/
│   ├── generate_topology.py
│   ├── train_ppo.py
│   ├── evaluate.py
│   ├── run_baselines.py
│   ├── run_ablation.py
│   └── run_sensitivity.py
│
├── tests/
│   ├── test_topology.py
│   ├── test_ksp.py
│   ├── test_resource.py
│   ├── test_queue.py
│   └── test_env.py
│
└── outputs/
    ├── checkpoints/
    ├── logs/
    ├── figures/
    └── results/
```

---

# 3. `configs/`：实验配置

所有实验参数统一放入YAML文件，不在代码中写死。

例如：

```yaml
simulation:
  num_slots: 500
  slot_duration: 1.0
  seed: 42

constellation:
  num_planes: 6
  sats_per_plane: 8
  altitude_km: 550
  inclination_deg: 53

task:
  arrival_rate: 4
  data_min_mbit: 0.5
  data_max_mbit: 5.0
  cycles_min: 500
  cycles_max: 1500
  deadline_min: 0.5
  deadline_max: 2.0

routing:
  max_compute_hops: 3
  k_paths: 3
  lookahead_slots: 3

compute:
  cpu_capacity: 5.0e10

network:
  link_capacity: 1.0e9
```

用途：

- `base.yaml`：默认参数；
- `small.yaml`：小星座调试；
- `medium.yaml`：主实验；
- `large.yaml`：扩展性实验；
- `ppo.yaml`：PPO学习率、batch、gamma等训练参数。

---

# 4. `data/`：离线数据

## `data/topology/`

保存提前生成的动态卫星拓扑：

```text
positions.npy
adjacency.npy
distances.npy
capacities.npy
```

## `data/tasks/`

可选。保存固定任务trace，用于保证不同算法测试公平。

## `data/cache/`

保存KSP或候选节点等缓存，减少训练重复计算。

---

# 5. `src/topology/`：卫星轨道和拓扑

## 5.1 `walker.py`

功能：

- 生成Walker-Delta星座；
- 设置轨道面数；
- 每轨卫星数量；
- 高度；
- 倾角；
- 相位关系。

输入：

```text
num_planes
sats_per_plane
altitude
inclination
phase_factor
```

输出：

```text
每颗卫星的初始轨道参数
```

第一版可以使用简化Walker模型，不必立即上真实TLE。

---

## 5.2 `orbit_generator.py`

功能：

根据星座参数生成：

\[
\mathbf q_s(t)
\]

即：

```python
positions[t, satellite_id, xyz]
```

后期如果想增强真实性，可将此模块替换为：

```text
SGP4 + TLE
```

其他代码无需修改。

---

## 5.3 `graph_builder.py`

功能：

根据：

```text
卫星位置
+
ISL连接规则
```

建立：

\[
G(t)
\]

主要过程：

```text
positions(t)
    ↓
判断哪些卫星可以建立ISL
    ↓
计算距离
    ↓
计算propagation delay
    ↓
设置link capacity
    ↓
生成NetworkX Graph
```

边属性建议包括：

```python
{
    "distance": ...,
    "prop_delay": ...,
    "capacity": ...
}
```

---

## 5.4 `topology_cache.py`

功能：

提前生成并保存：

\[
G(0),G(1),\ldots,G(T)
\]

避免RL训练过程中不断重复计算轨道。

这是非常重要的性能优化。

---

# 6. `src/network/`：通信链路与路由基础

## 6.1 `link_model.py`

负责实现通信公式。

主要函数：

```python
calc_distance()
calc_propagation_delay()
calc_transmission_delay()
calc_link_delay()
```

对应：

\[
T_{ij}^{prop}=\frac{d_{ij}}{c}
\]

\[
T_{u,ij}^{tx}=\frac{D_u}{r_{u,ij}}
\]

\[
T_{u,ij}^{link}=T_{u,ij}^{tx}+T_{ij}^{prop}
\]

所有通信时延公式统一放这里，环境不重复实现。

---

## 6.2 `path_model.py`

定义Path对象，保存：

```python
Path(
    nodes=[1, 3, 5],
    hops=2,
    total_distance=...,
    estimated_delay=...,
    bottleneck_capacity=...
)
```

这样后面candidate builder、RL、metrics都可以直接使用同一个Path结构。

---

## 6.3 `ksp.py`

功能：

为：

\[
source\rightarrow destination
\]

生成：

\[
K
\]

条候选路径。

第一版直接使用：

```python
networkx.shortest_simple_paths()
```

输入：

```text
graph
source
destination
K
```

输出：

```python
[
    [1, 2, 5],
    [1, 3, 4, 5],
    [1, 6, 5]
]
```

不需要自己重新实现Yen算法。

---

## 6.4 `feasibility.py`

该模块是模型中的重要部分。

负责：

\[
\boxed{\text{未来拓扑可行性检查}}
\]

输入：

```text
task
candidate path
current slot
future topology cache
estimated rate
deadline
```

主要过程：

```text
估计任务到达第1跳的时间
    ↓
检查未来对应时刻链路是否存在

估计到达第2跳的时间
    ↓
继续检查

...
```

同时检查：

\[
\hat T_{route}
+
\hat T_{compute}
\le
\tau
\]

输出建议：

```python
{
    "feasible": True,
    "route_estimated_delay": 0.18,
    "deadline_margin": 0.42,
    "topology_margin": ...
}
```

后续直接作为Feasibility Mask基础。

---

# 7. `src/tasks/`：任务模块

## 7.1 `task.py`

定义统一Task对象。

建议：

```python
@dataclass
class Task:
    task_id: int
    source_sat: int
    data_size: float
    cycles_per_bit: float
    deadline: float
    arrival_slot: int

    @property
    def total_cycles(self):
        return self.data_size * self.cycles_per_bit
```

对应：

\[
L_u=D_uC_u
\]

所有模块都使用同一Task对象。

---

## 7.2 `task_generator.py`

负责随机产生星上任务。

第一版：

\[
N(t)\sim Poisson(\lambda)
\]

然后：

\[
D_u\sim U(D_{min},D_{max})
\]

\[
C_u\sim U(C_{min},C_{max})
\]

\[
\tau_u\sim U(\tau_{min},\tau_{max})
\]

源卫星可以：

- 均匀随机；
- 设置热点卫星。

---

## 7.3 `task_queue.py`

用于管理当前时隙产生、待分配或未完成的任务列表。

第一版如果采用“任务到来立即决策”，该模块可以保持简单。

---

# 8. `src/compute/`：计算节点状态

## 8.1 `satellite_compute.py`

定义每颗卫星的计算能力和当前状态：

```python
SatelliteComputeState(
    max_cpu=...,
    queue_cycles=...,
    utilization=...
)
```

提供：

```python
get_available_cpu()
get_queue_delay()
get_utilization()
```

---

## 8.2 `compute_queue.py`

维护：

\[
Q_s(t)
\]

核心功能：

```python
process_workload()
add_workload()
get_queue_delay()
step()
```

更新：

\[
Q_s(t+1)
=
[Q_s(t)-F_s\Delta t]^+
+
L_s^{new}
\]

这是强化学习长期性的重要来源。

---

# 9. `src/resource/`：资源优化

## 9.1 `cpu_allocator.py`

输入：

```text
某计算卫星
+
该卫星当前分配到的任务
```

计算：

\[
f_{u,s}^{*}
=
F_s
\frac{\sqrt{D_uC_u}}
{\sum_v\sqrt{D_vC_v}}
\]

输出：

```python
{
    task_id_1: cpu_1,
    task_id_2: cpu_2,
    ...
}
```

并检查：

\[
\sum_uf_{u,s}\le F_s
\]

---

## 9.2 `link_allocator.py`

对于每条ISL：

1. 找所有经过该链路的任务；
2. 按闭式规则分配传输资源。

\[
r_{u,e}^{*}
=
R_e
\frac{\sqrt{D_u}}
{\sum_v\sqrt{D_v}}
\]

输出：

```python
{
    (task_id, edge): allocated_rate
}
```

---

## 9.3 `joint_allocator.py`

统一调用：

```text
CPU Allocator
+
Link Allocator
```

例如：

```python
allocation = allocate_resources(
    tasks=tasks,
    actions=selected_actions,
    graph=current_graph
)
```

最终输出：

```text
每个任务：
    route
    每条链路分配速率
    CPU allocation
    route delay
    compute delay
    total delay
```

强化学习完全不直接处理资源公式。

---

# 10. `src/routing/`：候选动作生成

## 10.1 `candidate_builder.py`

这是环境和RL之间的重要桥梁。

对一个Task执行：

### Step 1

根据：

\[
H_c
\]

寻找候选计算卫星。

### Step 2

对每颗候选卫星生成：

\[
K
\]

条KSP。

### Step 3

调用：

```python
feasibility.py
```

进行未来拓扑检查。

### Step 4

检查deadline。

最终输出：

```python
CandidateAction(
    compute_sat=...,
    path=...,
    features=...,
    feasible=True
)
```

---

## 10.2 `action_builder.py`

构造RL实际看到的动作列表：

```python
[
    LOCAL,
    (sat_3, path_1),
    (sat_3, path_2),
    (sat_6, path_1),
    ...
]
```

动作数量是动态的。

建议统一定义：

```python
@dataclass
class RoutingAction:
    task_id: int
    compute_sat: int
    path: list[int]
    is_local: bool
```

所有算法，无论heuristic还是PPO，最终都返回同一结构。

---

## 10.3 `route_cost.py`

计算启发式路由代价，例如：

\[
J(s,p)
=
\alpha T_p
+
\beta\frac{Q_s}{F_s}
+
\gamma\frac{L_u}{F_s}
\]

供Computing-aware heuristic baseline使用，也方便做candidate feature。

---

# 11. `src/env/`：强化学习环境

## 11.1 `leo_env.py`

整个系统的总协调器。

不要把所有数学公式都写进这里。

每个step大致：

```text
1. 读取G(t)
2. 更新卫星计算队列
3. 生成新任务
4. 为任务生成candidate actions
5. Agent选择动作(s,p)
6. 调用资源优化器
7. 计算真实route delay
8. 计算compute delay
9. 检查deadline / route success
10. 更新Q_s(t+1)
11. 计算reward
12. 记录metrics
13. 进入t+1
```

采用Gymnasium接口：

```python
obs, info = env.reset()

next_obs, reward, terminated, truncated, info = env.step(action)
```

注意：

> `leo_env.py` 不应该知道action是PPO、shortest path还是heuristic产生的。

---

## 11.2 `state_builder.py`

负责将环境状态转为模型输入。

节点特征：

```python
node_features[s] = [
    normalized_queue,
    cpu_capacity,
    cpu_utilization
]
```

边特征：

```python
edge_features[e] = [
    normalized_distance,
    available_capacity,
    future_link_feature
]
```

任务特征：

```python
task_feature = [
    data_size,
    cycles_per_bit,
    deadline
]
```

这样后面修改state不需要改环境。

---

## 11.3 `reward.py`

第一版：

\[
r_t
=
-\bar T
-\lambda_d\nu_{ddl}
-\lambda_f\nu_{route}
\]

函数形式：

```python
compute_reward(
    average_delay,
    deadline_violation_rate,
    route_failure_rate
)
```

第一阶段不要堆大量shaping reward。

---

## 11.4 `metrics.py`

每个episode记录：

```text
average_delay
deadline_violation_rate
task_success_rate
route_failure_rate
average_hops
average_queue
compute_load_balance
link_utilization
```

论文后期所有图都依赖这一模块。

---

# 12. `src/models/`：神经网络

## 12.1 `gat_encoder.py`

输入：

```text
node_features
edge_index
edge_features
```

输出：

\[
h_s^G
\]

即卫星节点embedding。

第一版：

```text
2层GAT
hidden_dim = 64
```

即可。

后期可增加edge feature机制。

---

## 12.2 `task_encoder.py`

输入：

\[
[D,C,\tau]
\]

输出任务embedding：

\[
h_u
\]

简单MLP即可：

```text
3 → 64 → 64
```

---

## 12.3 `path_encoder.py`

输入candidate path人工特征：

```text
hop count
estimated route delay
bottleneck capacity
future feasibility margin
destination queue
destination CPU
```

输出：

\[
h_p
\]

第一版MLP即可，不需要再上Path-GNN。

---

## 12.4 `candidate_scorer.py`

Actor的核心模块。

构造：

\[
z_{u,a}
=
[
h_u,
h_{source},
h_s,
h_p
]
\]

对每个candidate：

\[
q_{u,a}
=
MLP(z_{u,a})
\]

最后：

```python
probs = softmax(scores)
```

候选数可变，因此：

```text
当前5个candidate → 5个概率
下一时隙13个candidate → 13个概率
```

不改变网络结构。

---

## 12.5 `value_network.py`

PPO Critic。

可以使用：

```text
Graph Pooling
+
Task Embedding
+
Global Load Feature
→ MLP
→ V(s)
```

第一版保持简单。

---

# 13. `src/agents/`：PPO

## 13.1 `ppo_agent.py`

主要函数：

```python
choose_action()
evaluate_action()
update()
save()
load()
```

`choose_action()`输入：

```text
graph
task
candidate actions
mask
```

输出：

```text
selected_candidate_index
log_probability
value
```

---

## 13.2 `rollout_buffer.py`

保存：

```text
state
task
candidate representation
chosen action
reward
log_prob
value
done
```

然后计算：

\[
GAE
\]

注意候选动作数量变化，因此不能简单假设固定长度动作向量。

---

## 13.3 `trainer.py`

负责完整训练循环：

```text
for episode:
    reset environment
    collect rollout
    compute GAE
    PPO update
    save checkpoint
    log metrics
```

避免训练逻辑全部写进`train_ppo.py`。

---

# 14. `src/baselines/`：对比算法

## 14.1 `local_only.py`

所有任务在源卫星本地执行。

用途：

验证跨星协同本身是否有效。

---

## 14.2 `shortest_path.py`

只考虑网络通信代价。

可选：

```text
选择预计通信时延最小的目标卫星-路径
```

代表传统网络层策略。

---

## 14.3 `least_load.py`

选择：

\[
\arg\min_s
\frac{Q_s}{F_s}
\]

然后采用shortest path。

用于比较“只看计算负载”与联合计算感知策略。

---

## 14.4 `computing_aware.py`

使用启发式代价：

\[
J(s,p)
=
\alpha T_p
+
\beta\frac{Q_s}{F_s}
+
\gamma\frac{L_u}{F_s}
\]

选最小。

这是一个非常重要的强baseline。

---

## 14.5 `vanilla_ppo.py`

不用GAT。

直接将：

```text
route delay
hop count
destination queue
CPU
bottleneck capacity
deadline margin
```

等特征输入PPO。

用于验证Graph Encoder价值。

---

# 15. `src/evaluation/`：实验管理

## 15.1 `evaluator.py`

统一测试所有算法。

必须固定：

```text
seed
topology trace
task trace
```

保证公平。

输出：

```csv
algorithm,delay,success_rate,route_failure,...
```

---

## 15.2 `ablation.py`

自动运行：

```text
Full Model
w/o GAT
w/o Look-ahead
w/o Compute Queue
Equal Resource
```

---

## 15.3 `sensitivity.py`

自动扫描：

```text
task arrival rate
task data size
cycles per bit
CPU capacity
ISL capacity
satellite number
K
max hop
lookahead horizon
```

避免每次人工修改配置。

---

# 16. `src/utils/`

## `seed.py`

统一固定：

```python
random.seed()
np.random.seed()
torch.manual_seed()
```

保证实验可重复。

## `logger.py`

记录训练和评估日志。

## `io.py`

负责：

- 读取YAML；
- 保存JSON/CSV；
- 加载topology cache。

## `math_utils.py`

放：

- normalize；
- safe divide；
- moving average；
- 辅助数学函数。

---

# 17. `scripts/`：真正运行的入口

## `generate_topology.py`

生成：

```text
positions
adjacency
distance
capacity
future availability
```

运行：

```bash
python scripts/generate_topology.py
```

---

## `run_baselines.py`

先跑：

```text
Local
Shortest
Least-load
Computing-aware
```

这是整个工程最先应该完成的实验脚本之一。

---

## `train_ppo.py`

训练普通PPO或GAT-PPO。

---

## `evaluate.py`

加载训练好的checkpoint并统一评估。

---

## `run_ablation.py`

自动执行消融。

---

## `run_sensitivity.py`

自动进行敏感性实验。

---

# 18. `tests/`：单元测试

强烈建议写，否则资源和队列公式很容易出错。

## `test_topology.py`

检查：

- 图是否连通；
- 边连接规则是否正确；
- 距离是否合理。

## `test_ksp.py`

检查：

- 路径起点/终点；
- 无重复节点；
- K条路径排序。

## `test_resource.py`

必须验证：

\[
\sum_ur_{u,e}\le R_e^{max}
\]

\[
\sum_uf_{u,s}\le F_s^{max}
\]

## `test_queue.py`

检查：

\[
Q_s(t)\ge0
\]

以及队列更新是否正确。

## `test_env.py`

检查：

- reset；
- step；
- state shape；
- reward；
- episode结束条件。

---

# 19. 环境与算法必须严格分离

这是整个工程最重要的原则之一。

环境只能接受统一Action：

```python
action = RoutingAction(...)
next_obs = env.step(action)
```

对于不同算法：

```python
action = shortest_policy.select(obs)
env.step(action)
```

和：

```python
action = ppo_agent.select(obs)
env.step(action)
```

环境执行逻辑必须完全一样。

这样baseline才公平。

---

# 20. 推荐统一Action结构

```python
@dataclass
class RoutingAction:
    task_id: int
    compute_sat: int
    path: list[int]
    is_local: bool
```

所有算法都只需要返回这个对象。

资源分配器、环境、metrics都不需要知道动作来自哪种算法。

---

# 21. 推荐Observation结构

不要一开始全部flatten。

可以定义：

```text
Observation
├── graph
├── node_features
├── edge_features
├── task_features
├── candidates
├── candidate_features
└── feasibility_mask
```

传统heuristic可以直接读取结构化状态。

神经网络再自行编码。

---

# 22. 完整数据流

```text
config
   ↓
topology generator
   ↓
G(t)
   ↓
task generator
   ↓
Task
   ↓
candidate builder
   ↓
Compute Satellite + KSP
   ↓
future feasibility filter
   ↓
Candidate Action Set
   ↓
Agent
   ↓
(s,p)
   ↓
resource allocator
   ↓
r* + f*
   ↓
route delay + compute delay
   ↓
deadline / route success
   ↓
compute queue update
   ↓
reward
   ↓
next state
```

---

# 23. 开发顺序

不要第一步写GAT-PPO。

## Milestone 1：静态网络跑通

目标：

```text
一个Task
→ candidate satellite
→ route
→ compute
→ total delay
```

此时不需要RL。

---

## Milestone 2：多任务和资源竞争

加入：

- 多个任务；
- 多任务共享ISL；
- 多任务共享CPU；
- 闭式资源分配。

成功标准：

\[
\sum r\le R^{max}
\]

\[
\sum f\le F^{max}
\]

始终满足。

---

## Milestone 3：动态计算队列

加入：

\[
Q_s(t)
\]

成功标准：

- queue随任务调度变化；
- heavy load时queue增大；
- load降低时queue能够下降。

---

## Milestone 4：动态卫星拓扑

生成：

\[
G(t)\rightarrow G(t+1)
\]

成功标准：

- 边能正常变化；
- route failure能够被观测。

---

## Milestone 5：完成Baseline

必须至少完成：

```text
Local
Shortest
Least-load
Computing-aware
```

此时即使RL还没完成，实验平台已经成立。

---

## Milestone 6：普通MLP-PPO

先不用GAT。

使用candidate人工特征训练PPO。

目标：

验证RL pipeline能够收敛。

---

## Milestone 7：GAT-PPO

将网络状态编码升级为GAT。

---

## Milestone 8：Feasibility Mask

最后加入：

\[
G(t+1),...,G(t+H_f)
\]

做未来拓扑可行性筛选。

---

# 24. 第一阶段真正需要写的文件

完整目录虽然很多，但最初只需要：

```text
configs/base.yaml

src/topology/graph_builder.py

src/tasks/task.py
src/tasks/task_generator.py

src/compute/compute_queue.py

src/network/link_model.py
src/network/ksp.py

src/resource/cpu_allocator.py
src/resource/link_allocator.py

src/routing/candidate_builder.py

src/env/leo_env.py
src/env/metrics.py

src/baselines/local_only.py
src/baselines/shortest_path.py
src/baselines/least_load.py

scripts/run_baselines.py
```

先把这十几个文件跑通。

此阶段：

\[
\boxed{\text{不要写GAT，不要写PPO}}
\]

---

# 25. 第二阶段再增加

```text
src/network/feasibility.py

src/models/task_encoder.py
src/models/path_encoder.py

src/agents/ppo_agent.py
src/agents/rollout_buffer.py
src/agents/trainer.py

src/baselines/vanilla_ppo.py

scripts/train_ppo.py
```

---

# 26. 第三阶段再增加

```text
src/models/gat_encoder.py
src/models/candidate_scorer.py
src/models/value_network.py

src/evaluation/ablation.py
src/evaluation/sensitivity.py

scripts/run_ablation.py
scripts/run_sensitivity.py
```

---

# 27. 论文模块与代码模块对应关系

| 论文内容 | 工程代码 |
|---|---|
| LEO动态网络模型 | `topology/` |
| ISL通信模型 | `network/link_model.py` |
| 星上任务模型 | `tasks/` |
| 计算队列 | `compute/compute_queue.py` |
| 候选计算节点 | `routing/candidate_builder.py` |
| KSP路由 | `network/ksp.py` |
| 未来拓扑可行性 | `network/feasibility.py` |
| 高层动作 | `routing/action_builder.py` |
| 链路资源优化 | `resource/link_allocator.py` |
| CPU资源优化 | `resource/cpu_allocator.py` |
| MDP环境 | `env/leo_env.py` |
| 状态设计 | `env/state_builder.py` |
| Reward | `env/reward.py` |
| GAT编码 | `models/gat_encoder.py` |
| Candidate评分 | `models/candidate_scorer.py` |
| PPO | `agents/ppo_agent.py` |
| Baseline | `baselines/` |
| 消融实验 | `evaluation/ablation.py` |
| 敏感性实验 | `evaluation/sensitivity.py` |

---

# 28. 推荐技术栈

```text
Python
│
├── NumPy
├── SciPy
├── NetworkX
├── PyYAML
├── pandas
├── PyTorch
├── PyTorch Geometric
├── Gymnasium
└── matplotlib
```

轨道模块后期需要真实性时，再加入：

```text
sgp4
skyfield
```

第一版不建议使用复杂网络仿真器。

---

# 29. 风险控制

## 风险1：GAT训练不稳定

退化为：

\[
Candidate~Features+MLP-PPO
\]

论文仍然可以完成。

---

## 风险2：未来拓扑过于复杂

先用：

\[
H_p=\min_{e\in p}H_e
\]

简化路径寿命。

后期再升级到逐跳未来adjacency检查。

---

## 风险3：轨道模型太难

先生成规则周期动态图或简化Walker。

系统和算法跑通后再换真实Walker/TLE。

---

## 风险4：资源闭式解实现有问题

先用：

\[
\sqrt{workload}
\]

比例分配，只要满足资源约束即可。

---

## 风险5：RL迟迟不收敛

论文最少仍可保留：

- 静态/动态图环境；
- Computing-aware heuristic；
- Future feasibility；
- Structured resource allocation。

然后重新选择DQN/PPO等更简单模型。

---

# 30. 推荐的实际开发节奏

建议按如下顺序：

```text
第1周
静态Graph + Task + KSP + Delay

第2周
CPU Queue + Resource Allocator + Baselines

第3周
Dynamic Graph + Future Feasibility

第4周
MLP-PPO

第5周
GAT Encoder

第6周
GAT-PPO + Mask

第7周以后
调参 + 消融 + 敏感性 + 画图
```

不要求严格按周，只表示优先级。

---

# 31. 最终工程应达到的最小可用状态

即使GAT-PPO暂时表现一般，至少应保证整个项目能够完成：

```text
1. 自动生成动态LEO网络
2. 自动生成星上任务
3. 自动维护计算队列
4. 自动产生候选计算节点
5. 自动产生KSP候选路由
6. 自动检查路径未来可行性
7. 自动执行CPU/链路资源分配
8. 自动计算任务完成时延
9. 自动运行至少4种baseline
10. 自动输出delay/success/failure等指标
11. 自动进行不同负载与资源规模实验
```

当上述内容全部完成后，第二研究点即使不考虑最终神经网络细节，也已经拥有完整、可重复的仿真实验平台。

之后的GAT-PPO主要用于证明：

> 在动态网络、动态计算负载和动态候选路径条件下，学习策略可以比静态/启发式策略做出更优的长期计算路由决策。

