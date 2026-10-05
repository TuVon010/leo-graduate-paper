# System Model

We consider a computing-enabled low Earth orbit (LEO) satellite network in which satellites jointly provide inter-satellite data forwarding and onboard task execution. Let $\mathcal S=\{1,\ldots,S\}$ denote the satellite set. Each satellite carries a computing server with an effective processing capacity $F_s$ in CPU cycles per second. A task can be executed at its source satellite or forwarded through inter-satellite links (ISLs) to another computing satellite. The destination and forwarding path are selected jointly, since a short communication path may lead to a heavily loaded server, whereas a lightly loaded server may become difficult to reach as the topology evolves. This setting follows the general motivation of computing-aware satellite routing, while the execution and prediction models below specify the system considered in this work. [Computing-aware LEO routing](https://arxiv.org/abs/2211.08820)

The task-admission horizon is divided into $N$ slots of duration $\delta t$. The beginning of slot $n$ is $t_n=n\delta t$, where $n\in\{0,\ldots,N-1\}$. Routing decisions are made at slot boundaries, whereas transmission, propagation, and computation evolve in continuous time within each slot. Tasks unfinished at a boundary retain their remaining service requirements in subsequent slots. After the admission horizon, an additional drain period of at most $N_{\rm d}$ slots is allowed without new task arrivals. The maximum observation horizon is therefore $\bar T=(N+N_{\rm d})\delta t$.

We assume a logical controller has access to current topology, computing capacities, workload summaries, and newly arrived tasks. Short-term orbital snapshots are also available for route prediction, but future task arrivals are unknown. The controller is an abstraction of the decision mechanism; signaling overhead and state-reporting delays are outside the present model. Task inputs are assumed to be available at their source satellites. Ground access transmission, result delivery, and satellite energy consumption are not included in the cost considered here.

## A. Orbital Motion and Dynamic Topology

We use a circular two-body Walker-Delta constellation to generate satellite positions. Let $P$ denote the number of orbital planes, $J$ the number of satellites per plane, and $S=PJ$. For plane index $p\in\{0,\ldots,P-1\}$ and member index $j\in\{0,\ldots,J-1\}$, the right ascension and initial orbital phase are

$$
\Omega_p=\frac{2\pi p}{P},\qquad
\psi_{p,j}=\frac{2\pi j}{J}+\frac{2\pi F_{\rm W}p}{PJ},
$$

where $F_{\rm W}$ is the Walker phasing factor. Given orbital altitude $h$, Earth radius $R_{\rm E}$, and gravitational parameter $\mu_{\rm E}$, the orbital radius and angular speed are

$$
r_{\rm orb}=R_{\rm E}+h,\qquad
\nu_{\rm orb}=\sqrt{\frac{\mu_{\rm E}}{r_{\rm orb}^{3}}}.
$$

Let $i$ be the inclination and $\vartheta_{p,j}[n]=\psi_{p,j}+\nu_{\rm orb}t_n$. The Earth-centered inertial position of satellite $(p,j)$ is

$$
\mathbf q_{p,j}[n]=r_{\rm orb}
\begin{bmatrix}
\cos\Omega_p\cos\vartheta_{p,j}[n]-\sin\Omega_p\sin\vartheta_{p,j}[n]\cos i\\
\sin\Omega_p\cos\vartheta_{p,j}[n]+\cos\Omega_p\sin\vartheta_{p,j}[n]\cos i\\
\sin\vartheta_{p,j}[n]\sin i
\end{bmatrix}.
$$

Satellite trajectories are prescribed by orbital motion and are not optimization variables. This model is an idealized orbital approximation rather than a TLE-driven high-fidelity propagator.

The network in slot $n$ is represented by an undirected graph $\mathcal G[n]=(\mathcal S,\mathcal E[n])$. For satellites $s$ and $s'$, their distance is

$$
d_{s,s'}[n]=\lVert\mathbf q_s[n]-\mathbf q_{s'}[n]\rVert_2.
$$

An ISL is geometrically eligible only when its distance does not exceed $d_{\max}$ and its line segment clears the Earth with margin $h_{\rm clr}$:

$$
\min_{\alpha\in[0,1]}
\left\lVert\mathbf q_s[n]+\alpha\big(\mathbf q_{s'}[n]-\mathbf q_s[n]\big)\right\rVert_2
>R_{\rm E}+h_{\rm clr}.
$$

The link-selection rule considers neighboring satellites within each orbital plane and eligible pairs in adjacent planes. Inter-plane connections additionally satisfy a latitude limit. Eligible pairs are selected subject to a maximum node degree $d_{\rm deg}$, with closer inter-plane pairs preferred. Consequently, geometric visibility alone does not imply that an ISL is established. Let $a_{s,s'}^{\rm ISL}[n]\in\{0,1\}$ indicate the selected link, with $a_{s,s'}^{\rm ISL}[n]=a_{s',s}^{\rm ISL}[n]$ and

$$
\sum_{s'\ne s}a_{s,s'}^{\rm ISL}[n]\le d_{\rm deg},\qquad \forall s,n.
$$

The selected topology and its link attributes are held constant over $[t_n,t_{n+1})$. Connectivity is not enforced by adding links that violate the selection rule.

## B. Task and Computing-Destination Model

Let $\mathcal B_n$ denote the batch of tasks arriving at $t_n$, and let $\mathcal U=\bigcup_{n=0}^{N-1}\mathcal B_n$. Task $u\in\mathcal B_n$ is represented by

$$
\{s_u,t_u,D_u,C_u,\tau_u\},\qquad t_u=t_n,
$$

where $s_u$ is the source satellite, $D_u$ is the input size in bits, $C_u$ is the required CPU cycles per bit, and $\tau_u$ is the latency deadline relative to arrival. Its total computing workload is

$$
W_u=D_uC_u.
$$

Tasks are indivisible: each task is executed entirely at one computing satellite. Local execution refers to computation at $s_u$; remote execution requires forwarding the complete input to another satellite. Task splitting, intermediate computation, and execution migration are not considered.

In the stochastic workload model, the batch size is Poisson distributed with mean $\lambda_{\rm a}$ per slot. Spatial heterogeneity is represented by a hotspot set $\mathcal H\subseteq\mathcal S$. With probability $p_{\rm hot}$, a source is drawn uniformly from $\mathcal H$; otherwise it is drawn uniformly from $\mathcal S$. Hence,

$$
\Pr(s_u=s)=\frac{1-p_{\rm hot}}{S}
+\mathbf 1\{s\in\mathcal H\}\frac{p_{\rm hot}}{|\mathcal H|}.
$$

Task sizes, computing intensities, and deadlines are independently sampled from configured uniform distributions over bounded ranges. Their distributions and the hotspot mixture are experimental workload parameters.

To bound the decision space, the candidate computing set contains satellites within $H_{\rm c}$ shortest-path hops of the source in the current graph:

$$
\mathcal S_u[n]=\left\{s\in\mathcal S:
\operatorname{dist}_{\mathcal G[n]}(s_u,s)\le H_{\rm c}\right\}.
$$

For each remote destination, at most $K$ loop-free paths of at most $H_{\rm p}$ hops are retained. Paths are ranked using the reference communication cost

$$
\ell_e[n]=\frac{D_{\rm ref}}{R_e[n]}+\frac{d_e[n]}{c_0},
$$

where $D_{\rm ref}$ is a fixed reference input size, $R_e[n]$ is link capacity in bits per second, and $c_0$ is the speed of light. This ranking constructs a bounded candidate set; it does not enumerate every route optimal for every task size or congestion state.

Let $\mathcal C_u[n]$ denote the resulting destination-path candidates. Candidate $c$ contains a computing satellite $s(c)$ and a path $\pi(c)=(s_u,\ldots,s(c))$. The local candidate is $(s_u,(s_u))$ and has zero communication hops. The binary selection variable $x_{u,c}$ satisfies

$$
\sum_{c\in\mathcal C_u[n]}x_{u,c}=1,\qquad
x_{u,c}\in\{0,1\},\qquad \forall u\in\mathcal B_n.
$$

The selected destination $s_u^{\star}$ and path $\pi_u$ remain fixed throughout the task's execution.

## C. Inter-Satellite Communication Model

Each established ISL $e=\{s,s'\}$ provides an aggregate service capacity $R_e[n]$. The base model uses a common configured ISL capacity, while the formulation allows snapshot-dependent capacities. An unavailable link has zero capacity. Here, capacity denotes a data service rate in bits per second, rather than radio bandwidth in hertz.

The configured capacity may represent the effective budget assigned to the studied task class, rather than the hardware peak rate. Capacity reserved for other services can be abstracted by this budget; their traffic and the underlying physical-layer link budget are not explicitly simulated.

Let $\mathcal A_e^{\rm tx}(t)$ be the set of tasks currently transmitting over link $e$. Both forwarding directions share the same undirected-link budget. If $r_{u,e}(t)$ is the rate allocated to task $u$, then

$$
r_{u,e}(t)\ge0,\qquad
\sum_{u\in\mathcal A_e^{\rm tx}(t)}r_{u,e}(t)\le R_e[n(t)],
\qquad n(t)=\left\lfloor\frac{t}{\delta t}\right\rfloor.
$$

A task receives link service only while it is transmitting on its current hop. Propagating tasks do not consume transmission capacity.

Communication follows complete-input store-and-forward operation. For path $\pi_u=(s_{u,0},\ldots,s_{u,L_u})$, let $b_{u,\ell}$ and $e_{u,\ell}$ denote the transmission start and finish times of hop $\ell\in\{1,\ldots,L_u\}$. Each hop must transmit all $D_u$ bits, and its finish time satisfies

$$
e_{u,\ell}=\inf\left\{t\ge b_{u,\ell}:
\int_{b_{u,\ell}}^{t}r_{u,\{s_{u,\ell-1},s_{u,\ell}\}}(z)\,dz\ge D_u\right\}.
$$

The propagation duration is

$$
p_{u,\ell}=\frac{d_{\{s_{u,\ell-1},s_{u,\ell}\}}[n(e_{u,\ell}^{-})]}{c_0}.
$$

Thus, $b_{u,1}=t_u$ and $b_{u,\ell+1}=e_{u,\ell}+p_{u,\ell}$. The distance is taken from the final transmission-service snapshot. If transmission finishes exactly at a topology boundary, its completion is settled before applying the next snapshot. Once the input has been fully transmitted on a hop, subsequent disappearance of that link does not cancel propagation already in progress.

If the current hop becomes unavailable before transmission finishes, or is unavailable when that hop is entered, the task terminates with a route failure. The present model does not reroute failed tasks. Availability of every edge at admission therefore does not guarantee successful delivery along the complete path.

## D. Computing Service and Workload States

A remotely assigned task becomes eligible for CPU service only after the final-hop propagation is complete. Its computing arrival time is

$$
t_u^{\rm cpu}=\begin{cases}
t_u,&L_u=0,\\
e_{u,L_u}+p_{u,L_u},&L_u>0.
\end{cases}
$$

Let $\mathcal A_s^{\rm cpu}(t)$ denote the set of tasks whose inputs have arrived and whose computation at satellite $s$ remains unfinished. The allocated CPU rate $f_{s,u}(t)$ satisfies

$$
f_{s,u}(t)\ge0,\qquad
\sum_{u\in\mathcal A_s^{\rm cpu}(t)}f_{s,u}(t)\le F_s,
\qquad \forall s,t.
$$

The server uses processor sharing, so multiple eligible tasks can receive CPU service concurrently. Rates are zero for tasks outside the corresponding service set. For a task in the computing stage, its remaining workload $W_u^{\rm rem}(t)$ evolves as

$$
\frac{dW_u^{\rm rem}(t)}{dt}=-f_{s_u^{\star},u}(t),\qquad
W_u^{\rm rem}(t_u^{\rm cpu})=W_u.
$$

The delivered CPU workload and the committed in-flight workload are separately defined as

$$
Q_s(t)=\sum_{u\in\mathcal A_s^{\rm cpu}(t)}W_u^{\rm rem}(t),
$$

$$
\widetilde Q_s(t)=
\sum_{\substack{u\text{ in transmission or propagation}\\s_u^{\star}=s}}W_u.
$$

The in-flight workload captures future demand already assigned to a server, but it is not eligible for CPU allocation. In particular, $Q_s/F_s$ is a workload-based congestion indicator; it is not an additional first-come-first-served waiting time added to processor-sharing execution.

## E. Structured Communication and Computing Allocation

For a fixed active service set $\mathcal A$ with positive workloads $w_u$, consider the static resource-allocation subproblem

$$
\min_{\{z_u>0\}}\sum_{u\in\mathcal A}\frac{w_u}{z_u},
\qquad \text{s.t.}\quad\sum_{u\in\mathcal A}z_u\le Z_{\rm res}.
$$

Here, $Z_{\rm res}>0$ is the resource budget. Its solution is $z_u^{\star}=Z_{\rm res}\sqrt{w_u}/\sum_{v\in\mathcal A}\sqrt{w_v}$. We use this structure to allocate resources among the current service participants:

$$
f_{s,u}(t)=F_s
\frac{\sqrt{W_u}}{\sum_{v\in\mathcal A_s^{\rm cpu}(t)}\sqrt{W_v}},
\qquad u\in\mathcal A_s^{\rm cpu}(t),
$$

$$
r_{u,e}(t)=R_e[n(t)]
\frac{\sqrt{D_u}}{\sum_{v\in\mathcal A_e^{\rm tx}(t)}\sqrt{D_v}},
\qquad u\in\mathcal A_e^{\rm tx}(t).
$$

The allocation weights use original task workloads and input sizes. Remaining cycles and bits determine completion events. Allocations are recomputed when service membership or topology changes and are held constant between events. Empty service sets receive no allocation.

These expressions solve the stated fixed-set surrogate and always respect resource budgets. They are not a global optimum for the dynamic completion-time problem, in which the active sets change as tasks progress. All evaluated routing policies use the same allocation rule, allowing comparisons to isolate their destination-path decisions.

## F. Task Delay, Deadlines, and Terminal Outcomes

For a successfully completed task, the completion time is

$$
t_u^{\rm cmp}=\inf\left\{t\ge t_u^{\rm cpu}:
\int_{t_u^{\rm cpu}}^{t}f_{s_u^{\star},u}(z)\,dz\ge W_u\right\}.
$$

The actual task completion delay is

$$
\begin{aligned}
T_u&=t_u^{\rm cmp}-t_u\\
&=\sum_{\ell=1}^{L_u}\big(e_{u,\ell}-b_{u,\ell}+p_{u,\ell}\big)
+t_u^{\rm cmp}-t_u^{\rm cpu}.
\end{aligned}
$$

For local execution, the communication sum is zero. Competition is already reflected in the time-varying service rates and the resulting transmission and computation durations. No extra $Q_s/F_s$ term is added to this realized delay.

The absolute deadline is $t_u+\tau_u$. We define on-time success by

$$
I_u^{\rm suc}=\mathbf 1\{u\text{ is completed and }T_u\le\tau_u\}.
$$

A deadline violation is recorded once if an unfinished active task reaches its deadline. In the default model, such a task continues receiving service, allowing its eventual delay to be measured. Deadlines are therefore soft service requirements. A route failure occurring before the deadline is an unsuccessful outcome but does not automatically count as an observed deadline violation.

Tasks still active at $\bar T$ are censored. Let $\mathcal U_{\rm cmp}$ denote completed tasks and $\mathcal U_{\rm eval}$ the evaluated arrival cohort. Mean completion delay and on-time success rate are

$$
\overline T_{\rm cmp}=\frac{1}{|\mathcal U_{\rm cmp}|}
\sum_{u\in\mathcal U_{\rm cmp}}T_u,\qquad
\rho_{\rm suc}=\frac{1}{|\mathcal U_{\rm eval}|}
\sum_{u\in\mathcal U_{\rm eval}}I_u^{\rm suc},
$$

where $\mathcal U_{\rm cmp}\subseteq\mathcal U_{\rm eval}$. Mean completion delay is undefined when there are no completed tasks. It is reported together with completion, deadline-violation, route-failure, and censoring rates, since completed-task delay alone can favor a policy that fails difficult tasks.

When a warmup period is used, the arrival cohort is restricted to the common admission window after warmup, and those tasks are followed through the drain period. Resource utilization is measured over the same fixed admission window for every policy.

## G. Short-Term Route Prediction and Batch Competition

At slot $n$, the controller can inspect the current snapshot and up to $H$ future snapshots. For $H>0$, the prediction cache covers intervals up to

$$
t_n^{\rm cov}=\min\{n+H+1,N_{\rm trace}\}\delta t,
$$

where $N_{\rm trace}$ is the number of available snapshots. This expression accounts for the fact that each snapshot describes a full half-open slot interval. Setting $H=0$ disables future checking and retains current-snapshot estimates.

For candidate path $\pi(c)$, each hop initially uses reference rate $\eta_R R_e[n]$, where $\eta_R\in(0,1]$. The predicted transmission interval is checked against every covered snapshot that it intersects. If capacities decrease, the reference rate is reduced to $\eta_R$ times the minimum checked capacity, and the expanded interval is checked again until the rate stabilizes or an unavailable link is detected. For a hop without detected link loss, its prediction is

$$
\widehat t_{u,e}^{\rm tx}=\frac{D_u}{\widehat r_{u,e}},\qquad
\widehat p_e=\frac{d_e[n]}{c_0}.
$$

Hop start times are propagated recursively according to store-and-forward operation. The resulting route estimate is

$$
\widehat T_{u,c}^{\rm net}=
\sum_{e\in\pi(c)}\left(\widehat t_{u,e}^{\rm tx}+\widehat p_e\right).
$$

Only transmission intervals require continued link availability; propagation after the final transmitted bit is excluded from contact checks. Let $\phi_{u,c}$ indicate that no unavailable link was detected in the checked transmission intervals, and let $\chi_{u,c}$ indicate that all predicted transmission intervals were fully covered. Partial coverage is treated as unverified availability, rather than a certified feasible route. These predictions use orbital snapshots and a reference service rate, not future workload realizations.

To account for competition within a newly arrived batch, decisions are made sequentially in ascending relative-deadline order, with task identifier used to break ties. Previously selected tasks create virtual workload reservations. Before selecting task $u$, let

$$
V_s^{\rm cpu}(u)=\sum_{v\prec u}\mathbf 1\{s_v^{\star}=s\}W_v,
\qquad
V_e^{\rm tx}(u)=\sum_{v\prec u}\mathbf 1\{e\in\pi_v\}D_v.
$$

Existing transmission demand on link $e$ is

$$
B_e^{\rm tx}(t_n)=\sum_{v\in\mathcal A_e^{\rm tx}(t_n)}D_{v,e}^{\rm rem}(t_n),
$$

where $D_{v,e}^{\rm rem}$ denotes residual bits on the task's current hop. The reservation-aware completion proxy is

$$
\begin{aligned}
\widehat T_{u,c}={}&\widehat T_{u,c}^{\rm net}
+\frac{Q_{s(c)}(t_n)+\widetilde Q_{s(c)}(t_n)+V_{s(c)}^{\rm cpu}(u)+W_u}{F_{s(c)}}\\
&+\sum_{e\in\pi(c)}\frac{B_e^{\rm tx}(t_n)+V_e^{\rm tx}(u)}{R_e[n]}.
\end{aligned}
$$

The CPU and link workload terms are congestion proxies for decision making. They are not exact processor-sharing waiting times or physical admission reservations. All tasks in the batch are admitted together after the selections have been made.

With predictive masking enabled, a candidate is retained when no covered contact loss is detected and its delay proxy meets the deadline. If $\beta\in\{0,1\}$ controls whether unverified future intervals are allowed, the eligibility indicator for $H>0$ is

$$
m_{u,c}=\phi_{u,c}\,
\mathbf 1\{\chi_{u,c}=1\text{ or }\beta=1\}\,
\mathbf 1\{\widehat T_{u,c}\le\tau_u\}.
$$

The default setting allows unverified intervals and records their coverage status. For $H=0$, the future-coverage condition is omitted. Deadline filtering and predictive masking can be disabled separately for controlled comparisons. If every candidate is masked, the local candidate is restored as an explicit fallback. A fallback does not imply that the task can meet its deadline. Likewise, an eligible route may still fail or finish late because realized resource sharing differs from the reference prediction.

## H. Delay-and-Reliability Objective and Problem Formulation

Let $\mathcal A(t)$ denote all active tasks, including transmission, propagation, and computing stages. The slot-level holding cost is

$$
H_n=\int_{t_n}^{t_{n+1}}|\mathcal A(t)|\,dt.
$$

Let $M_n$ be the number of newly recorded deadline violations in slot $n$, and let $J_n$ be the number of newly recorded route failures. The per-slot cost and reinforcement-learning reward are

$$
C_n=\frac{H_n+\alpha_{\rm d}M_n+\alpha_{\rm f}J_n}{Z},
\qquad r_n=-C_n,
$$

where $\alpha_{\rm d},\alpha_{\rm f}\ge0$ are penalty coefficients with units of seconds, and $Z>0$ is a fixed normalization constant. The denominator is independent of the policy's number of completions. Thus, tasks contribute delay cost while they remain unfinished, including during slots with no new arrivals.

For task $u$, let $\widehat t_u^{\rm end}$ be its completion or failure time, or $\bar T$ if it is censored. The undiscounted holding cost satisfies

$$
\sum_{n=0}^{N+N_{\rm d}-1}H_n=
\sum_{u\in\mathcal U}\left(\widehat t_u^{\rm end}-t_u\right).
$$

The right-hand side is aggregate observed sojourn time. It equals aggregate completion delay when every task completes, but remains defined for failed and censored tasks. The two penalties discourage deadline violations and route failures. No additional terminal censoring penalty is assumed.

Let $X=\{x_{u,c}\}$ denote destination-path decisions, $\mathcal F=\{f_{s,u}(t)\}$ CPU allocations, and $\mathcal R=\{r_{u,e}(t)\}$ link allocations. Let $\Pi$ denote a causal routing policy and $\mathcal I_n$ its available information at slot $n$: current system state, the current task batch, and the permitted orbital prediction window. The general stochastic optimization problem is

$$
\begin{aligned}
\text{(P1)}:\quad
\min_{\Pi,\mathcal F,\mathcal R}\quad
&\mathbb E_{\Pi}\!\left[\sum_{n=0}^{N+N_{\rm d}-1}\gamma^n C_n\right]\\
\text{s.t.}\quad
&\sum_{c\in\mathcal C_u[n]}x_{u,c}=1,
\quad x_{u,c}\in\{0,1\},\quad \forall u\in\mathcal B_n,\\
&\pi(c)\text{ is a current-graph, loop-free candidate path from }s_u\text{ to }s(c),\\
&\sum_{u\in\mathcal A_s^{\rm cpu}(t)}f_{s,u}(t)\le F_s,
\quad f_{s,u}(t)\ge0,\quad \forall s,t,\\
&\sum_{u\in\mathcal A_e^{\rm tx}(t)}r_{u,e}(t)\le R_e[n(t)],
\quad r_{u,e}(t)\ge0,\quad \forall e,t,\\
&f_{s,u}(t)=0\text{ outside the eligible CPU stage},\\
&r_{u,e}(t)=0\text{ outside the current transmission stage},\\
&X_n\sim\Pi(\cdot\mid\mathcal I_n),\\
&\mathcal F\text{ and }\mathcal R\text{ use only information available at service time},\\
&\text{store-and-forward, service-progress, and terminal-event dynamics hold}.
\end{aligned}
$$

Here, $\gamma\in(0,1]$ is the discount factor. With $\gamma=1$, the objective combines aggregate observed sojourn time with reliability penalties. With $\gamma<1$, it is the discounted cost used for policy training and should not be identified with unweighted mean completion delay. If the system empties during the drain period, all subsequent costs are zero.

Problem (P1) is a dynamic mixed-integer optimization problem. A destination-path decision changes both network demand and future computing demand, while resource sharing changes hop completion times, subsequent service membership, and exposure to topology transitions. Predictive eligibility is therefore a decision aid rather than a hard guarantee that $T_u\le\tau_u$.

In the proposed implementation, resource allocations are restricted to the structured rules in Section E. Writing these rules as $\mathcal F=\Psi_F(X)$ and $\mathcal R=\Psi_R(X)$, the policy-learning problem is

$$
\text{(P2)}:\quad
\min_{\Pi}\ \mathbb E_{\Pi}\!\left[
\sum_{n=0}^{N+N_{\rm d}-1}\gamma^n
C_n\big(X,\Psi_F(X),\Psi_R(X)\big)\right],
$$

subject to the candidate, causality, and execution constraints of (P1). This restriction provides feasible resource allocations while concentrating learning on computing-aware destination-path selection. It motivates the proposed task-conditioned GAT-PPO method, which combines graph encoding, predictive candidate masking, and reservation-aware batch decisions. Graph attention and PPO provide the representation and policy-optimization components, respectively; the resource rules and route predictor remain explicit system mechanisms. [Graph attention networks](https://arxiv.org/abs/1710.10903), [Proximal policy optimization](https://arxiv.org/abs/1707.06347)

---

## 写作说明与代码对应（不属于论文正文）

上面的英文正文按当前工程的实际执行语义撰写，可作为论文 System Model 初稿。它没有沿用第一研究点的部分卸载、UAV 可控轨迹、能耗目标或任务优先级，因为这些机制尚未出现在第二研究点的实现中。

| 正文内容 | 当前工程对应 | 需要保留的建模边界 |
| --- | --- | --- |
| 圆轨道、ECI 坐标、动态选边 | `topology/walker.py`、`topology/graph_builder.py` | 理想化 Walker 场景，不称为真实 TLE/SGP4 轨道验证 |
| 批量任务、热点混合分布 | `tasks/task_generator.py` | 热点概率是混合权重，热点实际占比还含全网均匀抽样贡献 |
| 计算目标与完整路径候选 | `routing/candidate_builder.py`、`network/ksp.py` | 整任务卸载，参考大小用于路径排序，候选集有跳数和数量上限 |
| 存储转发、传播、CPU 服务、断链失败 | `env/event_engine.py` | 传播不占发送容量；数据完整到达后才能进入 CPU |
| CPU 与链路平方根分配 | `resource/cpu_allocator.py`、`resource/link_allocator.py` | 原始 workload 决定权重；固定活动集合的静态代理解，不是动态全局最优 |
| CPU 剩余量与在途承诺量 | `compute/compute_queue.py`、`env/state_builder.py` | $Q/F$ 仅用于估计，不能叠加到仿真完成时延 |
| 未来发送区间检查 | `network/feasibility.py` | `H=0` 禁用未来检查；默认允许覆盖不足的候选，但不能称其已获保证 |
| 批次预约、增强代价、mask 与本地回退 | `agents/features.py`、`agents/ppo_agent.py` | 预约是策略内部代理量，不是执行器的物理预留；回退可能仍然超期 |
| 持有成本、一次性违约/失败惩罚 | `env/reward.py` | 默认违约后继续服务；reward 不是仅对已完成任务取平均时延 |
| 完成时延、成功率、censor 与固定窗口 | `env/metrics.py`、`evaluation/` | 完成时延是条件统计，必须同时报告失败和截尾情况 |

当前 `configs/base.yaml` 的主要设置为：24 颗卫星，$\delta t=0.25$ s，CPU 容量 20–100 Gcycles/s，ISL 容量 1 Gbit/s，平均每时隙 12 个任务，热点混合权重 0.3，$H_{\rm c}=3$，$H_{\rm p}=4$，$K=3$，$\eta_R=0.5$，$H=8$，$\beta=1$。这些是目前的实验配置，不能仅凭代码设定将其表述为已经过文献或实测校准的系统参数。$H=8$ 表示读取当前及之后 8 个快照；名义前视长度为 2 s，而完整缓存区间最多覆盖从当前边界起的 2.25 s。

当前惩罚系数为 $\alpha_{\rm d}=\alpha_{\rm f}=2$ s，固定归一化量 $Z=12$，PPO 默认 $\gamma=0.995$。任务保持成本包含所有实际执行槽及 drain 槽，评估指标则按指定到达窗口统计；两者不应混写成同一个指标。

若后续希望声称“直接优化平均完成时延”“严格满足 deadline”或“全局联合优化资源”，需要先改变目标或约束并同步修改实现。现版本更准确的描述是：在统一结构化资源分配下，学习目的计算节点与路由的联合选择，以降低折扣后的任务持有成本及违约、断链惩罚。该目标本身不保证提出的方法优于所有基线，性能结论仍需正式多 seed 实验支持。

上述三个外部链接分别用于计算感知路由背景、GAT 和 PPO 的来源说明。具体公式、假设、预测窗口和代价函数以本项目实现为准；论文正式排版时可将链接替换为 BibTeX 引用。
