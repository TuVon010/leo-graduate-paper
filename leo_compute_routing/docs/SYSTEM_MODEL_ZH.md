# 系统模型

![LEO 卫星边缘计算系统](../paper_figures/system_model_zh.png)

本文考虑一个多低地球轨道（low Earth orbit，LEO）卫星边缘计算系统。卫星集合为 $\mathcal S=\{0,\ldots,S-1\}$，各卫星搭载有效处理能力为 $F_s$ CPU cycles/s 的星上服务器，并通过星间链路（inter-satellite link，ISL）交换任务输入。任务可在源卫星计算，也可卸载到其他卫星。本研究采用三层协同决策：图强化学习选择计算卫星，预测接触感知图路由确定到该卫星的传输路径，闭式优化分配实际链路和 CPU 资源。

任务到达阶段包含 $N$ 个时隙，时隙长度为 $\delta t$，起点为 $t_n=n\delta t$。卸载决策在时隙边界执行，任务传输和计算在时隙内按连续时间事件推进。到达阶段结束后，继续运行至多 $N_d$ 个排空时隙；未完成任务的剩余工作量跨时隙保留。逻辑控制器获取当前任务、网络与计算状态，以及长度受限的未来轨道快照，但不能获取未来任务到达。任务输入已位于源卫星；地面接入、结果回传、信令开销与星上能耗暂不计入。

## A. 卫星轨道与动态 ISL 拓扑

采用圆轨道二体运动下的 Walker-Delta 星座。设轨道面数为 $P$、每面卫星数为 $J$，有 $S=PJ$。对于面编号 $p$ 和面内编号 $j$，定义

$$
\Omega_p=\frac{2\pi p}{P},\qquad
\psi_{p,j}=\frac{2\pi j}{J}+\frac{2\pi F_Wp}{PJ},\qquad
r=R_E+h,\qquad \nu=\sqrt{\frac{\mu_E}{r^3}}.
$$

其中 $F_W$ 为 Walker 相位因子，$h$ 为轨道高度，$R_E$ 为地球半径，$\mu_E$ 为地球引力参数。令 $\vartheta_{p,j}[n]=\psi_{p,j}+\nu t_n$，倾角为 $i$，地心惯性坐标位置为

$$
\mathbf q_{p,j}[n]=r\begin{bmatrix}
\cos\Omega_p\cos\vartheta_{p,j}[n]-\sin\Omega_p\sin\vartheta_{p,j}[n]\cos i\\
\sin\Omega_p\cos\vartheta_{p,j}[n]+\cos\Omega_p\sin\vartheta_{p,j}[n]\cos i\\
\sin\vartheta_{p,j}[n]\sin i
\end{bmatrix}.
$$

相邻轨道面之间与同一轨道面相邻卫星之间允许建立 ISL。链路需满足最大距离、地球遮挡、终端度数以及跨面纬度限制。设 $d_{ij}[n]=\|\mathbf q_i[n]-\mathbf q_j[n]\|$，链路视距条件为

$$
\min_{\zeta\in[0,1]}\|\mathbf q_i[n]+\zeta(\mathbf q_j[n]-\mathbf q_i[n])\|>R_E+h_{\rm clr}.
$$

优先建立满足条件的面内相邻链路，再将符合条件的跨面链路按距离升序连接，且节点度数不超过 $d_{\max}$。得到当前图 $G[n]=(\mathcal S,\mathcal E[n])$；不补造链路来强制连通。这里的距离贪心是外生 ISL 连接规则，强化学习不改变轨道或连接规则。物理拓扑可预先缓存，但策略仅能看到当前图与规定前视窗口。

每条活动链路 $e=\{i,j\}$ 的任务业务速率预算为 $R_e[n]$，传播时延为 $d_{ij}[n]/c_0$，其中 $c_0$ 为光速。默认使用固定活动链路容量；距离影响连接与传播，而非额外假设无线信道衰落。本模型是轨道近似与业务资源预算模型，不是 TLE/SGP4 传播器或实际在轨硬件仿真。

## B. 任务生成与整任务卸载

时隙 $n$ 的任务批次记为 $\mathcal U[n]$。任务 $u$ 表示为

$$
u=(s_u,D_u,C_u,\tau_u,t_u),\qquad W_u=D_uC_u,\qquad t_u=t_n,
$$

其中 $s_u$ 为源卫星、$D_u$ 为输入 bit 数、$C_u$ 为 cycles/bit、$\tau_u$ 为相对截止期限，$W_u$ 为总计算量。每时隙到达数服从泊松分布；数据量、计算密度及期限分别在配置范围独立均匀采样。设热点集合为 $\mathcal H$，热点混合概率为 $p_h$，则

$$
\Pr(s_u=s)=\frac{1-p_h}{S}+\frac{p_h}{|\mathcal H|}\mathbf1\{s\in\mathcal H\}.
$$

因此热点任务份额包含均匀分量，不能将 $p_h$ 直接当作热点实际份额。各卫星 CPU 容量在每个回放 seed 下采样一次，整段实验保持不变。

定义 $a_{u,s}\in\{0,1\}$ 表示任务最终执行于卫星 $s$，满足

$$
\sum_{s\in\mathcal S}a_{u,s}=1.
$$

任务不可拆分，不设置卸载比例。PPO 的随机动作仅为请求的计算卫星 $\widetilde s_u\in\mathcal S$，**不包含路径、不枚举计算节点与路径组合**。选择计算卫星之后，独立图算法求解路径。源卫星也是一个可选计算节点；本地执行路径为 $(s_u)$。

## C. 接触窗口与预测接触感知图路由

控制器可读取当前和之后 $H$ 个拓扑快照，预测覆盖到 $t_n+(H+1)\delta t$，名义前视长度为 $H\delta t$。$H=0$ 表示仅使用当前快照，不读取后续拓扑。链路 $e$ 的接触窗口由连续可用快照合并得到，表示为

$$
\mathcal C_e[n]=\{[b_{e,k},d_{e,k})\}_k,\qquad
K_{e,k}=\int_{b_{e,k}}^{d_{e,k}}R_e(t)\,dt.
$$

窗口右端若正好位于预测边界，不能将其表述为已知实际断链时刻。默认不接受远程传输中超出预测覆盖的区间；允许未知区间的开关需明确标注。预测不使用未来业务到达。

在 PPO 输出 $\widetilde s_u$ 后，图路由器只搜索从 $s_u$ 到该节点的无环路径

$$
\pi_u=(v_0=s_u,v_1,\ldots,v_{h_u}=\widetilde s_u),\qquad h_u\le H_p.
$$

搜索基于当前图的链路，逐跳计算实际任务大小下的预计发送区间，检查未来快照是否仍有足够接触时间和容量。参考服务速率取 $\eta_R R_e(t)$，$\eta_R\in(0,1]$。设链路预测预约占用集合为 $\mathcal B_e$，则一跳需满足

$$
\int_{\widehat b_{u,e}}^{\widehat d_{u,e}}
\eta_R R_e(t)\mathbf1\{t\notin\mathcal B_e\}\,dt\ge D_u,
$$

且发送区间内链路连续可用。发送完成后经历传播时延，随后才开始下一跳；传播过程不要求前一条链路继续可用。本实现允许参考预约造成的发送延后，但**不跨越已知断链等待下一次接触，不进行执行中重路由**。

路径评分为

$$
J_{\rm route}(\pi_u)=\widehat T_u^{\rm net}(\pi_u)
+\lambda_r\sum_{e\in\pi_u}\frac{1}{1+m_{u,e}/t_{\rm ref}}
+\lambda_l\sum_{e\in\pi_u}\frac{B_e^{\rm tx}[n]}{D_u},
$$

其中 $m_{u,e}\ge0$ 为预测发送完成后剩余的已覆盖接触窗口长度，$t_{\rm ref}=1$ s，$B_e^{\rm tx}[n]$ 为当前正在该链路发送的任务剩余 bit 数。$\lambda_r,\lambda_l$ 的单位为 s；风险和拥塞项是路由偏好，不额外计入真实物理时延。无未来信息时风险项关闭。

图算法采用时间相关的有界路径标签搜索，返回扩展预算内已找到的最佳完整路径；不声称所有情形下的全局最优或精确 CGR。搜索被截断时单独记录。链路预测日历由现有活动任务和同批次此前决策建立，CPU 预约在数据预计到达后开始；预约仅用于预测，不改变真实资源共享。

PPO 输出前只做节点级基本筛选：当前图最短跳数不超过 $H_c$，并可选地剔除独占最快执行都无法满足期限的节点，即 $W_u/F_s>\tau_u$。该下界忽略传输和竞争，不是安全保证。所有节点被筛掉时恢复源节点并标记回退。图路由完成后再检查接触覆盖和预约完成时间。无可用路径或预测无法按期完成时，实际执行节点回退为 $s_u$；保留原请求节点与回退原因。本地任务仍可能超期。

## D. 存储转发与计算共享

链路采用整输入存储转发，每跳传输 $D_u$ bit。设 $\mathcal T_e(t)$ 为正在链路 $e$ 发送的任务集合，其分配速率为 $r_{u,e}(t)$，满足

$$
\sum_{u\in\mathcal T_e(t)}r_{u,e}(t)\le R_e(t),\qquad r_{u,e}(t)\ge0.
$$

同一条无向 ISL 的两个方向共享一个容量预算；不将两个方向各自计算为完整容量。任务只占用当前跳，未开始的后续跳不占实际链路资源。在发送阶段链路断开会终止任务并记录路由失败；预测拒绝与实际断链失败分开统计。

设 $\mathcal K_s(t)$ 为数据已到达且正在卫星 $s$ 计算的任务集合，分配 CPU 速率为 $f_{s,u}(t)$，满足

$$
\sum_{u\in\mathcal K_s(t)}f_{s,u}(t)\le F_s,\qquad f_{s,u}(t)\ge0.
$$

CPU 采用处理器共享，各活动任务持续获得服务。计算剩余量满足 $\dot w_u(t)=-f_{s,u}(t)$，当前跳剩余 bit 满足 $\dot d_u(t)=-r_{u,e}(t)$。只在数据完整到达计算卫星后启动 CPU 服务。

令 $Q_s(t)=\sum_{u\in\mathcal K_s(t)}w_u(t)$，在途承诺计算量为 $I_s(t)$。$Q_s/F_s$ 和 $I_s/F_s$ 用于负载特征与预测；不能在事件仿真完成时延上再叠加一个 $Q_s/F_s$ 排队项。该模型不是 FIFO 串行队列。

## E. KKT 闭式资源分配

固定一个当前活动集合，考虑静态代理子问题

$$
\min_{x_u>0}\sum_u\frac{w_u}{x_u},\qquad
\text{s.t.}\ \sum_ux_u\le C.
$$

由 KKT 条件 $-w_u/x_u^2+\zeta=0$ 得

$$
x_u^*=C\frac{\sqrt{w_u}}{\sum_v\sqrt{w_v}}.
$$

链路代入 $w_u=D_u,C=R_e(t)$；CPU 代入 $w_u=W_u,C=F_s$，得到

$$
r_{u,e}(t)=R_e(t)\frac{\sqrt{D_u}}{\sum_{v\in\mathcal T_e(t)}\sqrt{D_v}},\qquad
f_{s,u}(t)=F_s\frac{\sqrt{W_u}}{\sum_{v\in\mathcal K_s(t)}\sqrt{W_v}}.
$$

实现使用任务**原始**数据量和计算量作为权重；剩余量只决定完成事件。到达、发送结束、传播结束、计算完成或拓扑变化时重新分配。该解是固定活动集合静态代理问题的最优解，不能称为整个动态网络的联合全局最优。资源消融采用相同预算下的等分分配。

## F. 任务时延、违约与优化目标

对于完成任务，完成时延为

$$
T_u=t_u^{\rm done}-t_u
=\sum_{e\in\pi_u}T_{u,e}^{\rm tx}+\sum_{e\in\pi_u}T_{u,e}^{\rm prop}+T_u^{\rm cpu}.
$$

各阶段持续时间由实际动态共享资源决定，本地任务网络项为零。成功定义为完成且 $T_u\le\tau_u$。默认到期任务只记录一次违约并继续执行；排空结束仍未完成的任务记录为截尾，不伪造完成时间。

设 $A(t)$ 为尚未终止的活动任务数，时隙持有成本为

$$
H_n=\int_{t_n}^{t_{n+1}}A(t)\,dt.
$$

以 $M_n$ 表示本时隙新期限违约数，$J_n$ 表示实际断链失败数，$B_n$ 表示选定远程节点后路由被拒绝并回退的任务数。使用原路由惩罚系数同时约束拒绝行为，以避免策略无成本地反复请求不可执行节点：

$$
C_n=H_n+\alpha_dM_n+\alpha_f(J_n+B_n),\qquad r_n=-C_n/Z.
$$

其中 $\alpha_d,\alpha_f$ 单位为 s，$Z$ 为固定归一化常数。本地基本筛选回退本身不计作物理路由失败；$B_n$ 与 $J_n$ 分别保存。持有成本包含完成、断链终止、丢弃或截尾前的全部活动时间，不能称为仅对完成任务的平均时延。所有方法使用相同代价定义。

在确定性的图路由映射 $\mathcal R$ 与资源映射 $\Psi$ 下，分层优化问题写为

$$
\text{(P1)}:\quad\min_\theta\ \mathbb E_{\pi_\theta}\!\left[\sum_{n=0}^{N+N_d-1}\gamma^nC_n\right],
$$

$$
\widetilde s_u\sim\pi_\theta(\cdot\mid\mathcal O_n,u,\text{批次前缀}),\quad
(s_u^*,\pi_u)=\mathcal R(\mathcal O_n,u,\widetilde s_u),\quad
(r(t),f(t))=\Psi(\text{当前活动任务与资源预算}),
$$

并满足唯一计算节点、无环路径、跳数限制与通信/计算容量约束。$\gamma$ 为折扣系数。期限采用筛选与违约惩罚，不是所有任务必须满足的硬约束。PPO 优化节点决策的长期代价；图算法处理此刻到该节点的传输；闭式规则处理当前资源分配。训练有限且预测近似，不能预先保证策略优于所有启发式。

## G. 图强化学习状态与动作

图节点状态包括当前 CPU 剩余工作量、异构容量、在途计算量、归一化 ECI 坐标、源于该节点的新批次计算量及任务数。边状态包括距离、容量、当前发送积压和预测窗口可用比例。任务状态包括大小、计算密度、期限、独占执行尺度及批次位置。

GAT 得到节点表示 $\mathbf h_s$，图池化得到全局表示 $\mathbf h_G$。共享评分器对每一个卫星节点计算

$$
\ell_{u,s}=g_\theta(\mathbf h_s,\mathbf h_{s_u},\mathbf h_G,\mathbf h_u,\mathbf z_{u,s}),\qquad
\pi_\theta(\widetilde s_u=s\mid\mathcal O_n,u)=\frac{\exp(\ell_{u,s})\,m_{u,s}}{\sum_j\exp(\ell_{u,j})\,m_{u,j}},
$$

其中 $m_{u,s}$ 为节点基本筛选标记，$\mathbf z_{u,s}$ 包含 CPU、在途及批次前缀承诺量、独占计算时长、源到节点距离与当前最短跳数。策略不输入预先搜索好的路径、路径排名或手工完成时间 logit 先验。卫星 ID 只用于索引，不作为可学习数值特征；共享评分器的输出数量随星座节点数变化。

同批任务按期限、任务 ID 顺序决策。每次节点选择后执行路由预测，再更新实际执行节点的预测预约，供后续任务使用；整个批次提交后物理时间才前进。PPO 保存请求节点的条件概率、采样时特征和掩码，对一个物理时隙使用批次联合概率比，不将路由回退重新当作一次 PPO 采样。

MLP-PPO 保持相同节点评分和后续路由，仅替换图编码器。48→24/72/96 星评估冻结网络权重及训练特征尺度。可变输出结构支持跨规模运行，但不等于已经证明零样本性能优势。

## H. 参数、代码对应与论文表述边界

| 设置 | compute24 | contact66 |
|---|---|---|
| Walker 结构 | 3×8，24 星 | 6×11，66 星 |
| 高度 / 倾角 | 600 km / 53° | 780 km / 80° |
| ISL 业务容量 | 200 Mbit/s | 100 Mbit/s |
| CPU | 40–100 Gcycles/s | 40–100 Gcycles/s |
| 输入 / 计算密度 | 20–60 Mbit / 1000–2000 cycles/bit | 20–60 Mbit / 500–1500 cycles/bit |
| 到达强度 | 4/时隙，即 16/s | 11/时隙，即 44/s |
| 热点数量 / 混合概率 | 3 / 0.2 | 8 / 0.2 |
| 期限 | 2–8 s | 2–8 s |
| 时隙 / 到达时域 / 排空上限 | 0.25 s / 600 s / 300 s | 同左 |
| 前视 / 参考速率比例 | H=16 / 0.5 | 同左 |

两者是不同物理实验场景，使用同一套方法；contact66 不是“算法名称”。参数是明确的工程假设，未声称实测校准。热点份额分别约为 0.30 和 0.297，compute24 热点典型负载约为 $16\times0.1\times60/70=1.37$；分母是每星 70 Gcycles/s，不是将三颗星容量错误放大十倍。具体 CPU 随固定回放 seed 变化，应查看 `calibration.json`。

| 模型部分 | 实现 |
|---|---|
| 轨道、物理 ISL 与缓存 | `topology/walker.py`、`topology/graph_builder.py`、`topology/topology_cache.py` |
| 任务与异构 CPU | `tasks/task_generator.py`、`env/leo_env.py` |
| 只输出计算卫星的 PPO | `models/destination_scorer.py`、`agents/ppo_agent.py` |
| 有限未来接触与选定节点路由 | `network/contact_plan.py`、`routing/contact_aware_router.py` |
| 预测预约 | `routing/reservations.py` |
| 动态真实执行、回退记录 | `env/event_engine.py`、`routing/action_builder.py` |
| KKT 分配 | `resource/cpu_allocator.py`、`resource/link_allocator.py` |
| 代价、成功、断链与路由拒绝 | `env/reward.py`、`env/metrics.py` |

当前版本固定训练回放 seed 2026；验证与开发评估使用同一 seed 100。复用验证数据的比较不叫独立测试，单 seed 不产生置信区间。完成任务均值/P95 必须与成功率、实际断链率、路由拒绝率和截尾率共同报告。`node_greedy` 使用相同预测路由、预约和资源分配，是区分长期学习与当步贪心的对照。论文结论以正式训练和这些一致对照为依据。
