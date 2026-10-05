# 多LEO协同边缘计算中的计算感知路由与资源协同优化研究方案

> **用途**：硕士毕业论文第二研究点 / 独立小论文方案  
> **建议大论文题目**：面向非地面网络边缘计算的通信计算协同优化方法研究  
> **第二研究点暂定题目**：面向动态多LEO卫星边缘计算的计算感知路由与资源协同优化方法

---

## 1. 研究定位

本研究面向具有星间链路（Inter-Satellite Link, ISL）和星载边缘计算能力的多LEO卫星网络。源卫星产生计算任务后，任务既可以在本星执行，也可以通过一条或多条ISL转发至其他计算资源更充足的卫星执行。

研究的核心不是传统意义上的“数据如何最快传到目标节点”，而是：

> **一个计算任务应该在哪颗卫星执行、通过哪条星间路径到达该卫星，以及沿途通信资源和最终计算资源如何分配，才能使任务尽可能快且可靠地完成。**

因此，研究问题可以概括为：

\[
\boxed{
\text{计算节点选择}
+
\text{星间路由}
+
\text{通信资源分配}
+
\text{计算资源分配}
}
\]

其核心区别在于：

\[
\boxed{
\text{Shortest Communication Path}
\neq
\text{Shortest Task Completion Path}
}
\]

传统路由主要关心网络传输时延，而本文关注：

\[
\boxed{
T^{task}
=
T^{network}
+
T^{computing}
}
\]

---

## 2. 与现有研究的关系

本研究的基本模型尽量建立在成熟文献基础上，避免为了“创新”而构造不合理场景。

可重点参考以下几类工作：

1. **Computing-Aware Routing for LEO Satellite Networks: A Transmission and Computation Integration Approach**  
   主要借鉴：通信与计算联合建模、动态LEO网络中的计算感知路由思想。

2. **Delay-cost computation offloading for on-board emergency tasks in LEO Satellite Edge Computing networks**  
   主要借鉴：星上任务、本星计算与跨星卸载、多跳ISL、deadline、Walker-Delta星座。

3. **Computation offloading and resource allocation in satellite edge computing networks: A multi-agent reinforcement learning approach**  
   主要借鉴：跨时隙任务状态、计算队列、强化学习与资源分配。

4. **Fully-Distributed Dynamic Packet Routing for LEO Satellite Networks: A GNN-Enhanced Multi-Agent Reinforcement Learning Approach**  
   主要借鉴：GAT/GNN对动态卫星图的编码方式。

5. **DRL-assisted task offloading in enhanced time-expanded graph (eTEG)-modeled aerial computing**  
   主要借鉴：未来拓扑、时间展开思想、动态通信/计算资源建模。

6. **Cooperative path scheduling and resource allocation for LEO satellite-enabled mobile edge computing network**  
   主要作用：用于避免重复。该类工作已经联合考虑计算节点、路径、通信资源和计算资源，因此本文不能仅以“联合路由和资源优化”为主要创新。

因此，本研究拟重点形成以下差异化：

\[
\boxed{
\text{短时拓扑可行性感知}
+
\text{任务感知的可变动作Graph-PPO}
+
\text{强化学习与结构化资源优化分解}
}
\]

---

# 3. 系统场景

考虑一个由多颗具有星载边缘计算能力的LEO卫星构成的单层卫星计算网络。

示意如下：

```text
                         LEO 4
                    [Compute Server]
                       /      \
                      /        \
             LEO 2 ----------- LEO 5
              /  \                |
             /    \               |
Source LEO 1 ---- LEO 3 -------- LEO 6
    │                              │
    │                              │
  Task                         Compute
```

对于源卫星产生的任务，有两种基本处理方式。

### 3.1 本地计算

```text
Source LEO
    │
    └── Local Computing
```

### 3.2 跨卫星协同计算

```text
Source LEO
    │
    ▼
  LEO 2
    │
    ▼
  LEO 5
    │
    ▼
Compute
```

任务最终只由一颗计算卫星完整执行，第一版暂不考虑任务拆分、多卫星并行计算和执行过程中的服务迁移。

---

# 4. 基本假设

为保证研究具有足够完整性，同时控制毕业论文实现风险，采用以下假设：

1. 卫星采用单层LEO星座，可采用Walker-Delta模型生成；
2. 卫星轨迹由轨道决定，不属于优化变量；
3. 卫星间通过ISL连接；
4. 每颗卫星具有有限星载CPU资源；
5. 每个任务只由一颗卫星完整执行；
6. 任务可以在源卫星本地计算，也可以跨星卸载；
7. 第一版不考虑任务执行中途迁移；
8. 网络采用离散时隙模型；
9. 轨道信息可用于获得未来短时域拓扑；
10. 强化学习只负责离散的“计算节点+路径”决策；
11. 连续通信资源和计算资源由数学优化模块分配。

---

# 5. 动态LEO网络模型

定义卫星集合：

\[
\mathcal S=\{1,2,\ldots,S\}
\]

时间被划分为：

\[
\mathcal T=\{1,2,\ldots,T\}
\]

个时隙，每个时隙长度为：

\[
\Delta t
\]

在时隙 \(t\)，LEO网络表示为动态图：

\[
\boxed{
\mathcal G(t)
=
\left(
\mathcal S,
\mathcal E(t)
\right)
}
\]

其中：

- \(\mathcal S\)：卫星节点集合；
- \(\mathcal E(t)\)：时隙 \(t\) 可用的ISL集合。

卫星 \(s\) 在时隙 \(t\) 的位置为：

\[
\mathbf q_s(t)
=
[x_s(t),y_s(t),z_s(t)]^T
\]

位置由Walker模型、SGP4或TLE提前生成，不属于强化学习动作。

因此网络满足：

\[
\mathcal E(t)
\neq
\mathcal E(t+1)
\]

实际代码中可以提前生成：

```text
position[t][s]
adjacency[t][i][j]
distance[t][i][j]
link_capacity[t][i][j]
```

RL训练期间直接读取缓存，不重复计算轨道。

---

# 6. 星间通信模型

对于星间链路：

\[
e=(i,j)\in\mathcal E(t)
\]

卫星间距离为：

\[
\boxed{
d_{ij}(t)
=
\|
\mathbf q_i(t)-\mathbf q_j(t)
\|
}
\]

传播时延为：

\[
\boxed{
T_{ij}^{prop}(t)
=
\frac{d_{ij}(t)}{c}
}
\]

其中 \(c\) 为光速。

为避免把研究重点扩展到复杂物理层，第一版采用有效链路容量模型。

定义：

\[
R_{ij}^{max}(t)
\]

为链路 \((i,j)\) 的最大有效传输速率。

任务 \(u\) 在链路 \((i,j)\) 上获得：

\[
r_{u,ij}(t)
\]

的传输资源，满足：

\[
\boxed{
\sum_{u:(i,j)\in p_u}
r_{u,ij}(t)
\le
R_{ij}^{max}(t)
}
\]

任务输入数据量为 \(D_u(t)\)，则该链路的传输时延：

\[
\boxed{
T_{u,ij}^{tx}(t)
=
\frac{D_u(t)}
{r_{u,ij}(t)}
}
\]

单跳总通信时延：

\[
\boxed{
T_{u,ij}^{link}(t)
=
\frac{D_u(t)}
{r_{u,ij}(t)}
+
\frac{d_{ij}(t)}{c}
}
\]

---

# 7. 星上任务模型

设时隙 \(t\) 新产生的任务集合为：

\[
\mathcal U(t)
\]

任务 \(u\) 由源卫星：

\[
o_u\in\mathcal S
\]

产生。

任务表示为：

\[
\boxed{
\Phi_u(t)
=
\{
D_u(t),
C_u(t),
\tau_u(t)
\}
}
\]

其中：

- \(D_u(t)\)：输入数据量，bit；
- \(C_u(t)\)：计算强度，cycles/bit；
- \(\tau_u(t)\)：任务deadline。

任务总计算量：

\[
\boxed{
L_u(t)
=
D_u(t)C_u(t)
}
\]

该任务模型属于MEC/卫星边缘计算中的成熟模型，不作为核心创新。

---

# 8. 计算卫星选择模型

任务既可以在源卫星本地执行，也可以卸载到其他卫星。

定义：

\[
x_{u,s}(t)\in\{0,1\}
\]

其中：

\[
x_{u,s}(t)=1
\]

表示任务 \(u\) 最终在卫星 \(s\) 上执行。

每个任务只选择一个计算节点：

\[
\boxed{
\sum_{s\in\mathcal S_u^c(t)}
x_{u,s}(t)=1
}
\]

其中：

\[
\mathcal S_u^c(t)
\]

为候选计算卫星集合。

允许：

\[
s=o_u
\]

即源卫星本地计算。

---

# 9. 候选计算卫星集合

为避免动作空间随着星座规模快速增长，引入最大候选跳数：

\[
H_c
\]

定义：

\[
\boxed{
\mathcal S_u^c(t)
=
\{
s\in\mathcal S:
hop(o_u,s)\le H_c
\}
}
\]

第一版建议：

\[
H_c=2\sim4
\]

同时保证：

\[
o_u\in\mathcal S_u^c(t)
\]

候选集合限制主要用于控制算法复杂度，不作为核心创新。

---

# 10. K-shortest候选路径

对于：

\[
o_u\rightarrow s
\]

采用Yen K-shortest path等传统图算法生成：

\[
\boxed{
\mathcal P_{u,s}(t)
=
\{
p_{u,s}^{1},
p_{u,s}^{2},
\ldots,
p_{u,s}^{K}
\}
}
\]

建议：

\[
K=3\sim5
\]

例如：

\[
p_1=S_1-S_2-S_5
\]

\[
p_2=S_1-S_3-S_4-S_5
\]

不建议让RL逐跳选择next-hop，原因包括：

- 容易产生环路；
- 动作序列过长；
- credit assignment困难；
- 无效探索多；
- 训练不稳定。

因此采用：

\[
\boxed{
KSP候选路径
+
RL选择完整路径
}
\]

---

# 11. 路由时延模型

任务 \(u\) 选择路径：

\[
p_u
\]

后，星间路由总时延：

\[
\boxed{
T_{u,p}^{route}(t)
=
\sum_{(i,j)\in p_u}
\left[
\frac{D_u(t)}
{r_{u,ij}(t)}
+
\frac{d_{ij}(t)}{c}
\right]
}
\]

即：

\[
\text{Transmission Delay}
+
\text{Propagation Delay}
\]

---

# 12. 卫星计算资源模型

每颗卫星 \(s\) 具有最大计算能力：

\[
F_s^{max}
\]

系统为任务 \(u\) 分配：

\[
f_{u,s}(t)
\]

CPU资源。

满足：

\[
\boxed{
\sum_{u:s_u=s}
f_{u,s}(t)
\le
F_s^{max}
}
\]

任务执行时间：

\[
\boxed{
T_{u,s}^{exe}(t)
=
\frac{D_u(t)C_u(t)}
{f_{u,s}(t)}
}
\]

---

# 13. 计算队列模型

定义：

\[
\boxed{
Q_s(t)
}
\]

表示卫星 \(s\) 在时隙 \(t\) 开始时仍待处理的计算量，单位为CPU cycles。

采用轻量等待时间近似：

\[
\boxed{
T_s^{queue}(t)
=
\frac{Q_s(t)}
{F_s^{max}}
}
\]

因此任务总计算时延：

\[
\boxed{
T_{u,s}^{comp}(t)
=
T_s^{queue}(t)
+
\frac{D_u(t)C_u(t)}
{f_{u,s}(t)}
}
\]

计算队列更新：

\[
\boxed{
Q_s(t+1)
=
\left[
Q_s(t)
-
F_s^{max}\Delta t
\right]^+
+
\sum_{u:s_u=s}
D_u(t)C_u(t)
}
\]

其中：

\[
[x]^+=\max(x,0)
\]

该队列状态非常关键，因为它使当前决策影响未来状态：

\[
a_t\rightarrow s_{t+1}
\]

从而使该问题具备真正的长期MDP结构。

---

# 14. 任务端到端完成时延

若任务在本地执行：

\[
s_u=o_u
\]

则：

\[
\boxed{
T_u^{local}
=
T_{o_u}^{queue}
+
T_{u,o_u}^{exe}
}
\]

若任务跨星卸载，则：

\[
\boxed{
T_u(t)
=
T_{u,p}^{route}(t)
+
T_{s_u}^{queue}(t)
+
T_{u,s_u}^{exe}(t)
}
\]

因此：

\[
\boxed{
T_u
=
\text{星间传输}
+
\text{传播}
+
\text{计算等待}
+
\text{计算执行}
}
\]

整个研究关注的是：

\[
\boxed{
\text{End-to-End Task Completion Time}
}
\]

而不是单纯通信路径长度。

---

# 15. 短时未来拓扑可行性

这是本方案的一个主要差异化设计。

LEO轨道可预测，因此在时隙 \(t\) 不仅知道：

\[
\mathcal G(t)
\]

还可以提前获得：

\[
\mathcal G(t+1),
\mathcal G(t+2),
\ldots,
\mathcal G(t+H_f)
\]

已有工作已经通过snapshot-free network、TEG/eTEG等方法利用未来拓扑信息，因此“考虑未来拓扑”本身不是本文创新。

本文采用更轻量的实现方式：

> **不构建完整time-expanded graph，而利用已知轨道信息逐跳检查候选路径在任务预计经过该链路时是否仍然有效。**

---

# 16. 预测可行路径

设候选路径：

\[
p=(v_0,v_1,\ldots,v_K)
\]

其中：

\[
v_0=o_u,\qquad v_K=s
\]

根据参考传输速率估计任务到达第 \(k\) 条链路的时间：

\[
\hat t_{u,k}^{arr}
\]

转换为时隙偏移：

\[
\delta_{u,k}
=
\left\lceil
\frac{\hat t_{u,k}^{arr}}
{\Delta t}
\right\rceil
\]

对应链路：

\[
e_k=(v_{k-1},v_k)
\]

要求：

\[
\boxed{
a_{e_k}
\left(
t+\delta_{u,k}
\right)=1
}
\]

若路径上所有链路都满足该条件，则路径属于：

\[
\boxed{
\mathcal P_{u,s}^{fea}(t)
}
\]

即短时预测可行路径集合。

该机制可以理解为：

\[
\boxed{
Orbit-informed Short-Horizon Feasibility
}
\]

---

# 17. Deadline可行性

对于计算节点—路径组合：

\[
(s,p)
\]

估计任务完成时间：

\[
\hat T_{u,s,p}
\]

若：

\[
\hat T_{u,s,p}>\tau_u
\]

说明该方案大概率无法满足deadline。

因此最终动作集合：

\[
\boxed{
\mathcal A_u^{fea}(t)
=
\{
(s,p):
p\in\mathcal P_{u,s}^{fea}(t),
~
\hat T_{u,s,p}\le\tau_u
\}
}
\]

同时包含本地计算动作。

该可行性Mask用于减少明显无效探索，但Mask本身不单独作为核心创新。

---

# 18. 联合优化变量

最终考虑四类变量：

### 计算节点

\[
\mathcal X=\{x_{u,s}(t)\}
\]

### 路由

\[
\mathcal Y=\{y_{u,s,p}(t)\}
\]

### ISL传输资源

\[
\mathcal R=\{r_{u,ij}(t)\}
\]

### CPU计算资源

\[
\mathcal F=\{f_{u,s}(t)\}
\]

整个问题即：

\[
\boxed{
\text{Where to Compute}
+
\text{How to Route}
+
\text{Communication Resource}
+
\text{Computing Resource}
}
\]

---

# 19. 优化目标

第一版以长期平均任务完成时延为主要目标，同时惩罚deadline violation：

\[
\boxed{
\min
\limsup_{T\rightarrow\infty}
\frac1T
\sum_{t=1}^{T}
\mathbb E
\left[
\sum_{u\in\mathcal U(t)}
\left(
T_u(t)
+
\lambda_v
\mathbf 1
\{
T_u(t)>\tau_u(t)
\}
\right)
\right]
}
\]

第一版暂不加入：

- 能耗；
- 缓存；
- 功率控制；
- 服务迁移；
- DAG任务；
- 任务拆分。

---

# 20. 联合优化问题P1

\[
\begin{aligned}
(P1):
\min_{\mathcal X,\mathcal Y,\mathcal R,\mathcal F}
&~
\limsup_{T\rightarrow\infty}
\frac{1}{T}
\sum_{t=1}^{T}
\mathbb E
\left[
\sum_{u}
\left(
T_u(t)
+
\lambda_v
\mathbf 1
\{T_u(t)>\tau_u(t)\}
\right)
\right]
\\
\mathrm{s.t.}\quad
&
\sum_{s\in\mathcal S_u^c}
x_{u,s}(t)=1,
\\
&
\sum_{p\in\mathcal P_{u,s}^{fea}}
y_{u,s,p}(t)=x_{u,s}(t),
\\
&
\sum_{u:(i,j)\in p_u}
r_{u,ij}(t)
\le
R_{ij}^{max}(t),
\\
&
\sum_{u:s_u=s}
f_{u,s}(t)
\le
F_s^{max},
\\
&
x_{u,s}(t)\in\{0,1\},
\\
&
y_{u,s,p}(t)\in\{0,1\},
\\
&
r_{u,ij}(t)\ge0,
\\
&
f_{u,s}(t)\ge0.
\end{aligned}
\]

该问题属于动态混合离散—连续优化问题。

---

# 21. 问题分解

采用：

\[
\boxed{
\text{Graph RL}
+
\text{Structured Resource Optimization}
}
\]

### 上层强化学习

负责：

\[
\boxed{
(s,p)
}
\]

即：

- 选择计算卫星；
- 选择候选路径。

### 下层优化

负责：

\[
\boxed{
r_{u,e}
+
f_{u,s}
}
\]

即：

- 链路传输资源；
- 星载CPU资源。

这样避免RL同时输出大量离散和连续变量。

---

# 22. 通信资源分配

对于链路 \(e\)，设通过该链路的任务集合：

\[
\mathcal U_e
\]

求解：

\[
\min_{\{r_{u,e}\}}
\sum_{u\in\mathcal U_e}
\frac{D_u}{r_{u,e}}
\]

约束：

\[
\sum_{u\in\mathcal U_e}
r_{u,e}
\le
R_e^{max}
\]

其KKT形式可得到：

\[
\boxed{
r_{u,e}^{*}
=
R_e^{max}
\frac{\sqrt{D_u}}
{\sum_{v\in\mathcal U_e}\sqrt{D_v}}
}
\]

若加入任务权重：

\[
r_{u,e}^{*}
=
R_e^{max}
\frac{\sqrt{\omega_uD_u}}
{\sum_v\sqrt{\omega_vD_v}}
\]

---

# 23. CPU资源分配

对于卫星 \(s\)，任务集合：

\[
\mathcal U_s
\]

求解：

\[
\min_{\{f_{u,s}\}}
\sum_{u\in\mathcal U_s}
\frac{D_uC_u}{f_{u,s}}
\]

约束：

\[
\sum_{u\in\mathcal U_s}
f_{u,s}
\le
F_s^{max}
\]

可得到：

\[
\boxed{
f_{u,s}^{*}
=
F_s^{max}
\frac{\sqrt{D_uC_u}}
{\sum_{v\in\mathcal U_s}\sqrt{D_vC_v}}
}
\]

若加入权重：

\[
f_{u,s}^{*}
=
F_s^{max}
\frac{\sqrt{\omega_uD_uC_u}}
{\sum_v\sqrt{\omega_vD_vC_v}}
\]

---

# 24. 强化学习方法

建议第一版采用：

\[
\boxed{
Feasibility-Aware Task-aware GAT-PPO
}
\]

也可暂时简称：

\[
\boxed{
FA-GPPO
}
\]

最终算法名称等实验稳定后再确定。

---

# 25. 状态空间

系统状态主要由三部分组成：

\[
s_t=
\{
G_t,
\Phi_u,
Q(t)
\}
\]

### 节点特征

\[
\boxed{
h_s(t)
=
[
Q_s(t),
F_s^{max},
\rho_s(t)
]
}
\]

其中：

\[
\rho_s(t)
=
\frac{Q_s(t)}{F_s^{max}}
\]

### 边特征

\[
\boxed{
e_{ij}(t)
=
[
d_{ij}(t),
R_{ij}^{max}(t),
A_{ij}^{future}(t)
]
}
\]

### 任务特征

\[
\boxed{
g_u(t)
=
[
D_u,
C_u,
\tau_u,
o_u
]
}
\]

---

# 26. GAT图编码

采用GAT得到每颗卫星的图表示：

\[
\boxed{
h_s^G
=
GAT(G_t)
}
\]

GAT只是合理工具，不将“用了GAT”单独声称为创新。

---

# 27. 候选路径特征

对于候选：

\[
a=(s,p)
\]

提取：

\[
h_p=
[
hop(p),
\hat T_p^{route},
R_p^{bottleneck},
H_p^{future},
Q_s/F_s
]
\]

构造候选表示：

\[
\boxed{
z_{u,a}
=
[
h_{o_u}^{G},
h_s^{G},
h_p,
g_u
]
}
\]

---

# 28. 可变候选动作评分

由于：

\[
|\mathcal A_u^{fea}(t)|
\]

会随拓扑和任务变化，因此不采用固定长度动作输出层。

对每个候选动作统一评分：

\[
\boxed{
q_{u,a}
=
MLP(z_{u,a})
}
\]

随后：

\[
\boxed{
\pi(a|s)
=
\frac{\exp(q_{u,a})}
{\sum_{a'\in\mathcal A_u^{fea}}
\exp(q_{u,a'})}
}
\]

这样同一策略可以处理不同数量的候选计算节点和路径。

---

# 29. Feasibility Mask

若：

\[
a\notin\mathcal A_u^{fea}(t)
\]

则：

\[
\pi(a|s)=0
\]

该机制利用：

- 轨道未来可预测性；
- deadline；
- 路径传输时间；

减少明显不可行的探索。

---

# 30. Reward

第一版保持简单：

\[
\boxed{
r_t
=
-\bar T(t)
-
\lambda_d\nu^{ddl}(t)
-
\lambda_f\nu^{route}(t)
}
\]

其中：

- \(\bar T(t)\)：平均任务完成时延；
- \(\nu^{ddl}\)：deadline violation rate；
- \(\nu^{route}\)：route failure rate。

不要一开始加入大量shaping reward。

---

# 31. 拟创新点

## 创新点1：轻量短时拓扑可行性感知

不构建完整TEG/eTEG，而利用已知轨道未来状态，对KSP候选路径逐跳进行短时可行性判断。

目标：

- 降低路径中断率；
- 避免snapshot只看当前连通性的缺陷；
- 不显著增加状态和动作空间。

---

## 创新点2：任务感知的可变动作Graph-PPO

将：

\[
\boxed{
Compute~Satellite+Route
}
\]

作为统一candidate action。

使用：

\[
Task~Embedding
+
Graph~Embedding
+
Path~Embedding
\]

为动态数量候选统一评分。

创新重点不在“GAT本身”，而在：

\[
\boxed{
\text{动态图表示}
+
\text{Task-conditioned Candidate Scoring}
+
\text{Feasibility Mask}
}
\]

---

## 创新点3：学习与结构化资源优化分解

强化学习只处理高维动态离散决策：

\[
\text{Compute Node+Route}
\]

连续：

\[
\text{ISL Resource+CPU Resource}
\]

由凸优化/KKT求解。

优势：

- 减少RL动作维度；
- 提高训练稳定性；
- 保证资源约束；
- 增强可解释性。

---

# 32. 可选增强项

若前三项实验后创新性仍需加强，可进一步加入：

\[
\mu_s^{CPU}
\]

和：

\[
\mu_e^{Link}
\]

等资源优化问题的KKT dual variable，作为下一时隙Graph RL的节点/边资源稀缺特征。

形成：

\[
\boxed{
Learning
\leftrightarrow
Optimization
}
\]

这一项建议作为后续增强，不影响第一版落地。

---

# 33. 实验目标

实验需要回答：

1. 计算感知路由是否优于传统通信最短路？
2. Graph-PPO是否优于普通PPO？
3. 短时未来拓扑可行性是否能降低路由失败？
4. 资源优化是否优于平均资源分配？
5. 算法在不同任务负载、计算资源和通信资源下是否稳定？
6. 算法是否具备一定星座规模泛化能力？

---

# 34. 星座规模

建议：

| 场景 | 卫星数量 | 用途 |
|---|---:|---|
| Small | 24 | 调试 |
| Medium | 48 | 主实验 |
| Large | 72或96 | 可扩展性 |

轨道参数可优先参考Walker-Delta常见配置，例如：

- 高度：550–600 km；
- 倾角：53°–60°。

---

# 35. 任务参数建议

第一阶段可使用：

| 参数 | 初始范围 |
|---|---|
| \(D\) | 0.5–5 Mbit |
| \(C\) | 500–1500 cycles/bit |
| \(\tau\) | 0.5–2 s |
| \(F_s\) | 20–100 Gcycles/s |
| ISL容量 | 0.5–2 Gbit/s |
| \(H_c\) | 2–4 hops |
| \(K\) | 3–5 paths |
| Look-ahead | 0–10 slots |

最终参数需根据实际时延量级校准。

---

# 36. 任务到达模型

采用：

\[
N(t)\sim Poisson(\lambda)
\]

通过：

\[
\lambda
\]

控制：

- 低负载；
- 中负载；
- 高负载。

源卫星可以随机生成，也可以设置少数热点卫星。

---

# 37. Baseline设计

## Baseline 1：Local Only

所有任务都在源卫星处理。

---

## Baseline 2：Shortest Network Delay

只根据网络通信时延选择目标卫星和路径。

---

## Baseline 3：Least-Load + Shortest Path

先选择：

\[
\arg\min_s\frac{Q_s}{F_s}
\]

再采用shortest path。

---

## Baseline 4：Computing-Aware Heuristic

定义：

\[
J(s,p)
=
\alpha \hat T_p^{route}
+
\beta\frac{Q_s}{F_s}
+
\gamma\frac{L_u}{F_s}
\]

选择最小代价方案。

---

## Baseline 5：Vanilla PPO

不用GNN，直接输入候选人工特征。

---

## Baseline 6：GAT-PPO without Look-ahead

只看当前：

\[
G(t)
\]

不使用未来拓扑。

---

## Proposed

\[
\boxed{
Task-aware GAT-PPO
+
Predictive Feasibility
+
Structured Resource Optimization
}
\]

---

# 38. 小规模最优参考

对6–10颗卫星和少量任务的场景，可以枚举所有候选：

\[
(s,p)
\]

并调用资源优化器求解，得到Oracle/近似最优结果。

评价：

\[
\frac{J_{RL}-J_{Oracle}}
{J_{Oracle}}
\]

用于说明RL策略与小规模最优解之间的差距。

---

# 39. 消融实验

至少包括：

### w/o Future Topology

取消短时未来拓扑可行性。

### w/o GAT

GAT替换为MLP。

### w/o Compute Queue

只使用最大CPU能力，不考虑计算队列。

### Equal Resource

取消KKT资源优化，改为平均分配。

### 可选：w/o Variable Candidate Scoring

固定动作编码，用于验证可变候选动作建模价值。

---

# 40. 敏感性实验

主要扫描：

1. 任务到达率 \(\lambda\)；
2. 任务数据量 \(D\)；
3. 计算强度 \(C\)；
4. CPU能力 \(F_s^{max}\)；
5. ISL容量 \(R_{ij}^{max}\)；
6. 星座规模；
7. KSP数量 \(K\)；
8. 最大候选跳数 \(H_c\)；
9. Look-ahead Horizon \(H_f\)。

特别建议重点展示：

\[
H_f=0,1,3,5,10
\]

其中：

\[
H_f=0
\]

等价于普通snapshot决策。

---

# 41. 核心评价指标

### 平均任务完成时延

\[
\bar T
=
\frac1N
\sum_uT_u
\]

### Deadline Violation Rate

\[
\nu^{ddl}
=
\frac{
\sum_u\mathbf1(T_u>\tau_u)
}
N
\]

### Task Success Rate

\[
P^{success}
=
1-\nu^{ddl}
\]

若考虑route failure，则只有路由成功且按时完成才算成功。

### Route Failure Rate

\[
\nu^{route}
=
\frac{N_{route-fail}}
N
\]

### Computing Load Balance

例如：

\[
Var(\rho_1,\ldots,\rho_S)
\]

### Link Utilization

统计链路平均/峰值利用率。

### Training Convergence

Reward、delay随训练步数变化。

### Inference Time

统计单次任务决策时间。

---

# 42. 推荐图表

建议最终至少包含：

1. PPO训练收敛曲线；
2. 各算法平均任务时延；
3. 各算法deadline violation；
4. 各算法task success rate；
5. 各算法route failure rate；
6. 不同任务到达率；
7. 不同任务大小；
8. 不同CPU能力；
9. 不同ISL capacity；
10. 不同look-ahead horizon；
11. 不同星座规模；
12. 消融实验；
13. inference time；
14. 可选：计算负载热力图；
15. 可选：实际路由示意图。

---

# 43. 实施顺序

## 阶段1：静态图

先跑通：

\[
Task
\rightarrow
Compute~Satellite
\rightarrow
Route
\rightarrow
Delay
\]

## 阶段2：多任务和资源竞争

加入：

- 共享链路；
- 共享CPU；
- KKT资源分配。

## 阶段3：动态计算队列

实现：

\[
Q_s(t)\rightarrow Q_s(t+1)
\]

## 阶段4：动态卫星拓扑

生成：

\[
G(1),G(2),\ldots,G(T)
\]

## 阶段5：Baseline

先完成：

- Local；
- Shortest；
- Least-load；
- Computing-aware heuristic。

## 阶段6：普通PPO

先不用GAT。

## 阶段7：GAT-PPO

再加入图编码。

## 阶段8：Future Feasibility Mask

最后加入未来拓扑机制。

---

# 44. 风险控制

### GAT-PPO不好训练

退化到：

\[
Candidate~Feature+PPO
\]

仍可完成论文。

### 动态拓扑实现困难

先做周期性动态图，而不是复杂packet-level动态网络。

### Future Look-ahead困难

先用简化路径剩余寿命：

\[
H_p=\min_{e\in p}H_e
\]

跑通，再升级逐跳检查。

### 资源推导复杂

可以直接使用平方根比例分配，保证资源约束满足。

### 创新被已有论文覆盖

不要声称：

- 首次计算感知路由；
- 首次GNN卫星路由；
- 首次动态LEO；
- 首次RL卫星卸载；
- 首次联合通信计算资源。

重点始终保持在：

\[
\boxed{
\text{轻量未来拓扑可行性}
+
\text{任务感知可变动作Graph-PPO}
+
\text{RL与资源优化分解}
}
\]

---

# 45. 与第一研究点的关系

建议大论文题目：

> **面向非地面网络边缘计算的通信计算协同优化方法研究**

### 第一研究点

**UAV辅助边缘计算中的任务卸载与资源协同优化**

核心：

\[
\boxed{
\text{接入侧任务卸载}
}
\]

主要变量：

- 用户关联；
- 部分卸载；
- UAV轨迹；
- CPU资源。

方法：

\[
CORA\text{-}MADDPG
\]

### 第二研究点

**多LEO协同边缘计算中的计算感知路由与资源协同优化**

核心：

\[
\boxed{
\text{网络侧计算感知路由}
}
\]

主要变量：

- 本地/跨星计算；
- 计算卫星；
- 多跳ISL；
- 链路资源；
- CPU资源。

方法：

\[
\boxed{
Graph~RL
+
Feasibility
+
Structured~Optimization
}
\]

因此整个大论文统一在：

\[
\boxed{
\text{非地面边缘计算中的通信—计算协同优化}
}
\]

这一主线上。

---

# 46. 当前方案一句话总结

> 本研究针对具有动态拓扑和有限星载计算资源的多LEO卫星边缘计算网络，考虑星上时延敏感任务的本地与跨星协同执行，通过短时轨道拓扑信息构造可行的“计算节点—路径”候选集合，利用任务感知Graph-PPO完成动态计算节点和多跳路由选择，并通过结构化优化完成ISL通信资源和星载CPU资源配置，从而降低长期任务完成时延并提高任务按时完成成功率。

