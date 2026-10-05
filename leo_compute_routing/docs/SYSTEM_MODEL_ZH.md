# 系统模型

本文考虑一个具备星上计算能力的低地球轨道（low Earth orbit，LEO）卫星网络，卫星同时承担星间数据转发和任务计算服务。设卫星集合为 $\mathcal S=\{1,\ldots,S\}$，每颗卫星搭载计算服务器，其有效处理能力为 $F_s$，单位为 CPU cycles/s。任务既可以在源卫星执行，也可以通过星间链路（inter-satellite link，ISL）转发至其他计算卫星。由于较短的通信路径可能通向负载较高的服务器，而负载较低的服务器又可能随拓扑演化变得难以到达，因此需要联合选择计算目的节点和转发路径。这一场景与计算感知卫星路由的研究动机一致，下文进一步给出本文采用的任务执行与预测模型。[计算感知 LEO 路由研究](https://arxiv.org/abs/2211.08820)

将任务到达阶段划分为 $N$ 个时隙，每个时隙持续 $\delta t$。第 $n$ 个时隙的起始时刻为 $t_n=n\delta t$，其中 $n\in\{0,\ldots,N-1\}$。路由决策在时隙边界作出，而传输、传播和计算过程在时隙内按连续时间演化。未完成任务的剩余服务需求保留至后续时隙，不在时隙边界重置。任务到达阶段结束后，系统继续运行至多 $N_{\rm d}$ 个时隙，以处理尚未完成的任务，该阶段不再产生新任务。因此，最大观测时域为 $\bar T=(N+N_{\rm d})\delta t$。

假设一个逻辑控制器能够获得当前网络拓扑、计算容量、工作量摘要和新到达任务，并可读取短期轨道快照以预测路由可行性，但不能获取未来任务到达信息。该控制器用于抽象决策机制，信令开销和状态上报时延不纳入当前模型。任务输入数据假设已位于源卫星，本文的代价模型不包含地面接入传输、结果回传和卫星能耗。

## A. 轨道运动与动态拓扑模型

采用基于圆轨道二体运动的 Walker-Delta 星座生成卫星位置。设轨道面数量为 $P$，每个轨道面的卫星数量为 $J$，则 $S=PJ$。对于轨道面索引 $p\in\{0,\ldots,P-1\}$ 和面内卫星索引 $j\in\{0,\ldots,J-1\}$，其升交点赤经与初始轨道相位分别为

$$
\Omega_p=\frac{2\pi p}{P},\qquad
\psi_{p,j}=\frac{2\pi j}{J}+\frac{2\pi F_{\rm W}p}{PJ},
$$

其中，$F_{\rm W}$ 为 Walker 相位因子。给定轨道高度 $h$、地球半径 $R_{\rm E}$ 和地球引力参数 $\mu_{\rm E}$，轨道半径与角速度为

$$
r_{\rm orb}=R_{\rm E}+h,\qquad
\nu_{\rm orb}=\sqrt{\frac{\mu_{\rm E}}{r_{\rm orb}^{3}}}.
$$

设轨道倾角为 $i$，并定义 $\vartheta_{p,j}[n]=\psi_{p,j}+\nu_{\rm orb}t_n$，则卫星 $(p,j)$ 在地心惯性坐标系中的位置为

$$
\mathbf q_{p,j}[n]=r_{\rm orb}
\begin{bmatrix}
\cos\Omega_p\cos\vartheta_{p,j}[n]-\sin\Omega_p\sin\vartheta_{p,j}[n]\cos i\\
\sin\Omega_p\cos\vartheta_{p,j}[n]+\cos\Omega_p\sin\vartheta_{p,j}[n]\cos i\\
\sin\vartheta_{p,j}[n]\sin i
\end{bmatrix}.
$$

卫星轨迹由轨道运动预先确定，不作为优化变量。该模型属于理想化轨道近似，不采用两行轨道根数（TLE）驱动的高精度轨道传播。

第 $n$ 个时隙的网络表示为无向图 $\mathcal G[n]=(\mathcal S,\mathcal E[n])$。卫星 $s$ 与 $s'$ 之间的距离为

$$
d_{s,s'}[n]=\lVert\mathbf q_s[n]-\mathbf q_{s'}[n]\rVert_2.
$$

只有当星间距离不超过 $d_{\max}$，且连接两颗卫星的线段与地球之间保持净空裕度 $h_{\rm clr}$ 时，该卫星对才具备建立 ISL 的几何条件：

$$
\min_{\alpha\in[0,1]}
\left\lVert\mathbf q_s[n]+\alpha\big(\mathbf q_{s'}[n]-\mathbf q_s[n]\big)\right\rVert_2
>R_{\rm E}+h_{\rm clr}.
$$

链路选择考虑同一轨道面内相邻卫星，以及相邻轨道面之间满足条件的卫星对。跨轨道面链路还需满足纬度限制。在最大节点度数 $d_{\rm deg}$ 的约束下，优先选择距离较近的跨轨道面卫星对。因此，几何可见并不意味着一定建立链路。令 $a_{s,s'}^{\rm ISL}[n]\in\{0,1\}$ 表示对应 ISL 是否建立，满足 $a_{s,s'}^{\rm ISL}[n]=a_{s',s}^{\rm ISL}[n]$，且

$$
\sum_{s'\ne s}a_{s,s'}^{\rm ISL}[n]\le d_{\rm deg},\qquad \forall s,n.
$$

选定的拓扑及链路属性在区间 $[t_n,t_{n+1})$ 内保持不变。系统不通过添加违反链路选择规则的连接来强制保证全网连通。

## B. 任务与计算目的节点模型

设 $\mathcal B_n$ 为时刻 $t_n$ 到达的任务批次，全部任务集合为 $\mathcal U=\bigcup_{n=0}^{N-1}\mathcal B_n$。对于任务 $u\in\mathcal B_n$，其属性表示为

$$
\{s_u,t_u,D_u,C_u,\tau_u\},\qquad t_u=t_n,
$$

其中，$s_u$ 为源卫星，$D_u$ 为输入数据量，单位为 bit；$C_u$ 为处理每比特数据所需的 CPU 周期数；$\tau_u$ 为相对于到达时刻的时延期限。任务的总计算工作量为

$$
W_u=D_uC_u.
$$

任务不可拆分，每个任务完整地在一颗计算卫星上执行。本地执行指在源卫星 $s_u$ 上计算，远程执行则需要将完整输入数据转发至另一颗卫星。当前模型不考虑任务拆分、中间节点计算和执行迁移。

在随机任务模型中，每个时隙的任务数量服从均值为 $\lambda_{\rm a}$ 的泊松分布。采用热点集合 $\mathcal H\subseteq\mathcal S$ 描述空间负载差异：以概率 $p_{\rm hot}$ 从 $\mathcal H$ 中均匀选择源卫星，否则从全部卫星集合 $\mathcal S$ 中均匀选择。因此，源卫星分布为

$$
\Pr(s_u=s)=\frac{1-p_{\rm hot}}{S}
+\mathbf 1\{s\in\mathcal H\}\frac{p_{\rm hot}}{|\mathcal H|}.
$$

任务数据量、计算强度和时延期限分别从给定有界区间上的均匀分布独立采样。这些分布及热点混合比例属于实验负载参数。

为控制决策空间规模，候选计算节点集合限定为当前图中距源卫星最短路径跳数不超过 $H_{\rm c}$ 的卫星：

$$
\mathcal S_u[n]=\left\{s\in\mathcal S:
\operatorname{dist}_{\mathcal G[n]}(s_u,s)\le H_{\rm c}\right\}.
$$

对于每个远程目的节点，最多保留 $K$ 条跳数不超过 $H_{\rm p}$ 的无环路径。路径按各链路参考通信代价之和排序，其中链路参考代价为

$$
\ell_e[n]=\frac{D_{\rm ref}}{R_e[n]}+\frac{d_e[n]}{c_0},
$$

其中，$D_{\rm ref}$ 为固定参考输入数据量，$R_e[n]$ 为链路容量，单位为 bit/s，$c_0$ 为光速。该排序用于构建有界候选集合，并不枚举所有任务大小和拥塞状态下的最优路径。

设 $\mathcal C_u[n]$ 为生成的“计算目的节点—路径”候选集合。候选 $c$ 包含计算节点 $s(c)$ 及路径 $\pi(c)=(s_u,\ldots,s(c))$。本地候选为 $(s_u,(s_u))$，其通信跳数为零。二进制选择变量 $x_{u,c}$ 满足

$$
\sum_{c\in\mathcal C_u[n]}x_{u,c}=1,\qquad
x_{u,c}\in\{0,1\},\qquad \forall u\in\mathcal B_n.
$$

选定的计算目的节点 $s_u^{\star}$ 和路径 $\pi_u$ 在任务执行期间保持不变。

## C. 星间通信模型

每条已建立的 ISL $e=\{s,s'\}$ 提供总服务容量 $R_e[n]$。基础模型采用统一配置的 ISL 容量，同时允许在一般形式下使用随快照变化的容量。不可用链路的容量为零。这里的容量表示单位时间内可传输的数据量，单位为 bit/s，而非单位为 Hz 的无线带宽。

配置容量也可以表示分配给所研究任务业务的有效预算，而非硬件峰值速率。其他业务预留的容量可通过该预算进行抽象，其实际流量和底层物理链路预算不在当前模型中显式仿真。

设 $\mathcal A_e^{\rm tx}(t)$ 为时刻 $t$ 正在链路 $e$ 上传输的任务集合。同一无向链路的两个转发方向共享一个容量预算。若分配给任务 $u$ 的传输速率为 $r_{u,e}(t)$，则

$$
r_{u,e}(t)\ge0,\qquad
\sum_{u\in\mathcal A_e^{\rm tx}(t)}r_{u,e}(t)\le R_e[n(t)],
\qquad n(t)=\left\lfloor\frac{t}{\delta t}\right\rfloor.
$$

任务仅在当前跳的发送阶段获得链路服务，处于传播阶段的任务不占用发送容量。

通信采用完整输入的存储转发机制。对于路径 $\pi_u=(s_{u,0},\ldots,s_{u,L_u})$，令 $b_{u,\ell}$ 和 $e_{u,\ell}$ 分别表示第 $\ell\in\{1,\ldots,L_u\}$ 跳的发送开始和结束时刻。每一跳均需发送全部 $D_u$ bit 输入数据，其发送结束时刻满足

$$
e_{u,\ell}=\inf\left\{t\ge b_{u,\ell}:
\int_{b_{u,\ell}}^{t}r_{u,\{s_{u,\ell-1},s_{u,\ell}\}}(z)\,dz\ge D_u\right\}.
$$

对应的传播时延为

$$
p_{u,\ell}=\frac{d_{\{s_{u,\ell-1},s_{u,\ell}\}}[n(e_{u,\ell}^{-})]}{c_0}.
$$

因此，$b_{u,1}=t_u$，且 $b_{u,\ell+1}=e_{u,\ell}+p_{u,\ell}$。传播距离取发送完成前最后一个服务区间所对应的拓扑快照。若发送恰好在拓扑边界完成，则先结算发送完成事件，再应用下一快照。一旦某跳已完整发送输入数据，随后该链路消失不会取消已经开始的传播过程。

若当前跳在发送完成前变为不可用，或任务进入该跳时链路已不可用，则任务以路由失败结束。当前模型不对失败任务重新路由。因此，任务到达时路径上的所有边均可用，并不能保证完整路径最终能够成功交付数据。

## D. 计算服务与工作量状态模型

远程任务只有在最后一跳传播完成后，才能获得 CPU 服务。其进入计算阶段的时刻为

$$
t_u^{\rm cpu}=\begin{cases}
t_u,&L_u=0,\\
e_{u,L_u}+p_{u,L_u},&L_u>0.
\end{cases}
$$

设 $\mathcal A_s^{\rm cpu}(t)$ 为输入已到达且尚未在卫星 $s$ 上完成计算的任务集合。分配给任务 $u$ 的 CPU 处理速率 $f_{s,u}(t)$ 满足

$$
f_{s,u}(t)\ge0,\qquad
\sum_{u\in\mathcal A_s^{\rm cpu}(t)}f_{s,u}(t)\le F_s,
\qquad \forall s,t.
$$

服务器采用处理器共享机制，多个符合服务条件的任务可以同时获得 CPU 服务。对于不属于对应服务集合的任务，其服务速率为零。处于计算阶段的任务，其剩余工作量 $W_u^{\rm rem}(t)$ 按下式演化：

$$
\frac{dW_u^{\rm rem}(t)}{dt}=-f_{s_u^{\star},u}(t),\qquad
W_u^{\rm rem}(t_u^{\rm cpu})=W_u.
$$

分别定义已送达的 CPU 剩余工作量和已承诺的在途工作量为

$$
Q_s(t)=\sum_{u\in\mathcal A_s^{\rm cpu}(t)}W_u^{\rm rem}(t),
$$

$$
\widetilde Q_s(t)=
\sum_{\substack{u\text{ 处于发送或传播阶段}\\s_u^{\star}=s}}W_u.
$$

在途工作量反映已经分配给服务器的未来计算需求，但不参与当前 CPU 分配。特别地，$Q_s/F_s$ 是基于工作量的拥塞指标，不是需要额外叠加到处理器共享执行时延上的先到先服务等待时间。

## E. 结构化通信与计算资源分配

对于固定的活动服务集合 $\mathcal A$，设各任务的工作量 $w_u>0$，考虑以下静态资源分配子问题：

$$
\min_{\{z_u>0\}}\sum_{u\in\mathcal A}\frac{w_u}{z_u},
\qquad \text{s.t.}\quad\sum_{u\in\mathcal A}z_u\le Z_{\rm res}.
$$

其中，$Z_{\rm res}>0$ 为资源预算。该子问题的解为 $z_u^{\star}=Z_{\rm res}\sqrt{w_u}/\sum_{v\in\mathcal A}\sqrt{w_v}$。据此，对当前参与服务的任务采用以下资源分配规则：

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

分配权重由任务原始计算工作量和原始输入数据量确定，剩余 CPU 周期数与剩余比特数用于判断完成事件。当服务成员或拓扑发生变化时，重新计算资源分配；相邻事件之间的服务速率保持不变。对于空服务集合，不分配资源。

上述表达式求解的是固定服务集合下的静态代理问题，并始终满足资源预算约束。由于任务推进会改变活动服务集合，它们不构成动态任务完成时延问题的全局最优解。各路由策略使用同一资源分配规则，从而使比较能够体现计算目的节点与路径选择的影响。

## F. 任务时延、期限与终止状态模型

对于已完成的任务，其完成时刻为

$$
t_u^{\rm cmp}=\inf\left\{t\ge t_u^{\rm cpu}:
\int_{t_u^{\rm cpu}}^{t}f_{s_u^{\star},u}(z)\,dz\ge W_u\right\}.
$$

任务的实际完成时延为

$$
\begin{aligned}
T_u&=t_u^{\rm cmp}-t_u\\
&=\sum_{\ell=1}^{L_u}\big(e_{u,\ell}-b_{u,\ell}+p_{u,\ell}\big)
+t_u^{\rm cmp}-t_u^{\rm cpu}.
\end{aligned}
$$

对于本地执行任务，通信时延之和为零。资源竞争已经通过时变服务速率反映在实际传输时长和计算时长中，因此不再向实际完成时延额外添加 $Q_s/F_s$ 项。

任务的绝对期限为 $t_u+\tau_u$。定义按时成功完成指示量为

$$
I_u^{\rm suc}=\mathbf 1\{u\text{ 已完成且 }T_u\le\tau_u\}.
$$

当仍处于活动状态的未完成任务达到期限时，记录一次期限违约。在默认模型中，该任务继续获得服务，以便测量最终完成时延。因此，期限属于软服务要求。若任务在期限之前因路由失败而终止，则属于未成功完成，但不会自动计为已经发生的期限违约。

在 $\bar T$ 时仍然活跃的任务标记为观测截尾（censored）。设 $\mathcal U_{\rm cmp}$ 为已完成任务集合，$\mathcal U_{\rm eval}$ 为纳入评估的到达任务集合，则平均完成时延和按时成功率分别为

$$
\overline T_{\rm cmp}=\frac{1}{|\mathcal U_{\rm cmp}|}
\sum_{u\in\mathcal U_{\rm cmp}}T_u,\qquad
\rho_{\rm suc}=\frac{1}{|\mathcal U_{\rm eval}|}
\sum_{u\in\mathcal U_{\rm eval}}I_u^{\rm suc},
$$

其中，$\mathcal U_{\rm cmp}\subseteq\mathcal U_{\rm eval}$。若没有任务完成，则平均完成时延未定义。评估时同时报告完成率、期限违约率、路由失败率和截尾率，避免仅统计已完成任务的时延而使容易放弃困难任务的策略获得表面优势。

采用预热阶段时，评估任务限定为预热结束后共同到达窗口内产生的任务，并继续跟踪其在后续排空阶段的执行结果。各策略的资源利用率均在相同的固定任务到达窗口内测量。

## G. 短期路由预测与批次竞争模型

在第 $n$ 个时隙，控制器能够读取当前快照以及至多 $H$ 个未来快照。当 $H>0$ 时，预测缓存覆盖区间的结束时刻为

$$
t_n^{\rm cov}=\min\{n+H+1,N_{\rm trace}\}\delta t,
$$

其中，$N_{\rm trace}$ 为可用快照数量。该表达式考虑了每个快照均描述一个完整的半开时隙区间。当 $H=0$ 时，禁用未来检查，仅保留基于当前快照的估计。

对于候选路径 $\pi(c)$，每一跳首先采用参考速率 $\eta_R R_e[n]$，其中 $\eta_R\in(0,1]$。对预测发送区间相交的每个已覆盖快照检查链路状态。若链路容量下降，则将参考速率降低至已检查最小容量的 $\eta_R$ 倍，并重新检查延长后的发送区间，直至速率稳定或检测到不可用链路。对于未检测到断链的跳，其预测发送时长与传播时延为

$$
\widehat t_{u,e}^{\rm tx}=\frac{D_u}{\widehat r_{u,e}},\qquad
\widehat p_e=\frac{d_e[n]}{c_0}.
$$

按照存储转发机制递推各跳的开始时刻，得到路径通信时延估计

$$
\widehat T_{u,c}^{\rm net}=
\sum_{e\in\pi(c)}\left(\widehat t_{u,e}^{\rm tx}+\widehat p_e\right).
$$

仅发送区间要求链路持续可用，最后一个比特发送完成后的传播阶段不纳入接触可用性检查。令 $\phi_{u,c}$ 表示已检查发送区间内未检测到不可用链路，$\chi_{u,c}$ 表示全部预测发送区间均得到完整覆盖。仅得到部分覆盖时，剩余区间的可用性记为尚未验证，不能视为已证明整条路由可行。预测依据轨道快照和参考服务速率，不使用未来实际负载。

为考虑新到达批次内的资源竞争，按照相对期限升序逐个作出决策，期限相同时按任务标识排序。已经选定的任务形成虚拟工作量预约。在为任务 $u$ 选择候选前，定义

$$
V_s^{\rm cpu}(u)=\sum_{v\prec u}\mathbf 1\{s_v^{\star}=s\}W_v,
\qquad
V_e^{\rm tx}(u)=\sum_{v\prec u}\mathbf 1\{e\in\pi_v\}D_v.
$$

其中，$v\prec u$ 表示同一批次中已先于 $u$ 完成选择的任务。链路 $e$ 上的现有发送需求为

$$
B_e^{\rm tx}(t_n)=\sum_{v\in\mathcal A_e^{\rm tx}(t_n)}D_{v,e}^{\rm rem}(t_n),
$$

其中，$D_{v,e}^{\rm rem}$ 为任务在当前跳上的剩余比特数。考虑预约的完成时延代理量为

$$
\begin{aligned}
\widehat T_{u,c}={}&\widehat T_{u,c}^{\rm net}
+\frac{Q_{s(c)}(t_n)+\widetilde Q_{s(c)}(t_n)+V_{s(c)}^{\rm cpu}(u)+W_u}{F_{s(c)}}\\
&+\sum_{e\in\pi(c)}\frac{B_e^{\rm tx}(t_n)+V_e^{\rm tx}(u)}{R_e[n]}.
\end{aligned}
$$

CPU 和链路工作量项用于提供决策所需的拥塞代理，它们不是处理器共享下的精确等待时间，也不构成物理资源预留。批次中的全部任务在完成选择后统一进入执行系统。

启用预测掩码时，只有在已覆盖区间未检测到断链，且时延代理量满足期限的候选才被保留。设 $\beta\in\{0,1\}$ 控制是否允许未来区间未完全验证的候选，则当 $H>0$ 时，候选保留指示量为

$$
m_{u,c}=\phi_{u,c}\,
\mathbf 1\{\chi_{u,c}=1\text{ 或 }\beta=1\}\,
\mathbf 1\{\widehat T_{u,c}\le\tau_u\}.
$$

默认设置允许未验证区间，同时保留覆盖状态信息。当 $H=0$ 时，省略未来覆盖条件。期限筛选与预测掩码可以分别关闭，以进行受控对比。若全部候选均被屏蔽，则恢复本地候选作为显式回退。该回退不意味着任务能够满足期限。同样，由于实际资源共享速率可能与参考预测不同，保留的路由仍可能失败或超期。

## H. 时延与可靠性目标及问题定义

设 $\mathcal A(t)$ 为全部活动任务集合，包含发送、传播和计算阶段。时隙级任务持有成本定义为

$$
H_n=\int_{t_n}^{t_{n+1}}|\mathcal A(t)|\,dt.
$$

令 $M_n$ 为第 $n$ 个时隙新增的期限违约数量，$J_n$ 为新增的路由失败数量。每时隙代价与强化学习奖励为

$$
C_n=\frac{H_n+\alpha_{\rm d}M_n+\alpha_{\rm f}J_n}{Z},
\qquad r_n=-C_n,
$$

其中，$\alpha_{\rm d},\alpha_{\rm f}\ge0$ 为单位为秒的惩罚系数，$Z>0$ 为固定归一化常数。分母不随策略在该时隙内的完成任务数量变化。因此，只要任务仍未终止，就持续产生时延成本，包括没有新任务到达的时隙。

对于任务 $u$，令 $\widehat t_u^{\rm end}$ 为其完成或失败时刻；若被截尾，则取 $\bar T$。未折扣的任务持有成本满足

$$
\sum_{n=0}^{N+N_{\rm d}-1}H_n=
\sum_{u\in\mathcal U}\left(\widehat t_u^{\rm end}-t_u\right).
$$

右侧为全部任务的累计观测逗留时间。当所有任务均完成时，它等于累计完成时延；对于失败和截尾任务，该量仍然有定义。两类惩罚分别用于抑制期限违约和路由失败，当前模型不额外设置终止时的截尾惩罚。

设 $X=\{x_{u,c}\}$ 为计算目的节点与路径决策，$\mathcal F=\{f_{s,u}(t)\}$ 为 CPU 分配，$\mathcal R=\{r_{u,e}(t)\}$ 为链路分配。令 $\Pi$ 为因果路由策略，$\mathcal I_n$ 为第 $n$ 个时隙可获得的信息，包含当前系统状态、当前任务批次及允许读取的轨道预测窗口。一般随机优化问题表示为

$$
\begin{aligned}
\text{(P1)}:\quad
\min_{\Pi,\mathcal F,\mathcal R}\quad
&\mathbb E_{\Pi}\!\left[\sum_{n=0}^{N+N_{\rm d}-1}\gamma^n C_n\right]\\
\text{s.t.}\quad
&\sum_{c\in\mathcal C_u[n]}x_{u,c}=1,
\quad x_{u,c}\in\{0,1\},\quad \forall u\in\mathcal B_n,\\
&\pi(c)\text{ 为当前图中从 }s_u\text{ 到 }s(c)\text{ 的无环候选路径},\\
&\sum_{u\in\mathcal A_s^{\rm cpu}(t)}f_{s,u}(t)\le F_s,
\quad f_{s,u}(t)\ge0,\quad \forall s,t,\\
&\sum_{u\in\mathcal A_e^{\rm tx}(t)}r_{u,e}(t)\le R_e[n(t)],
\quad r_{u,e}(t)\ge0,\quad \forall e,t,\\
&f_{s,u}(t)=0\text{，若任务不处于对应的可服务 CPU 阶段},\\
&r_{u,e}(t)=0\text{，若任务不处于对应链路的当前发送阶段},\\
&X_n\sim\Pi(\cdot\mid\mathcal I_n),\\
&\mathcal F\text{ 与 }\mathcal R\text{ 仅使用服务时刻已可获得的信息},\\
&\text{满足存储转发、服务进度和终止事件的状态演化关系}.
\end{aligned}
$$

其中，$\gamma\in(0,1]$ 为折扣因子。当 $\gamma=1$ 时，目标组合累计观测逗留时间与可靠性惩罚；当 $\gamma<1$ 时，目标为策略训练采用的折扣代价，不能直接等同于未加权平均完成时延。若系统在排空阶段提前清空，则后续时隙的代价均为零。

问题 (P1) 是一个动态混合整数优化问题。计算目的节点与路径选择同时改变网络需求和未来计算需求，资源共享又改变各跳完成时刻、后续服务集合，以及任务经历拓扑变化的情况。因此，预测候选筛选属于决策辅助机制，并不能硬性保证 $T_u\le\tau_u$。

在当前方法实现中，资源分配限定为 E 节给出的结构化规则。将两类规则表示为 $\mathcal F=\Psi_F(X)$ 和 $\mathcal R=\Psi_R(X)$，则策略学习问题为

$$
\text{(P2)}:\quad
\min_{\Pi}\ \mathbb E_{\Pi}\!\left[
\sum_{n=0}^{N+N_{\rm d}-1}\gamma^n
C_n\big(X,\Psi_F(X),\Psi_R(X)\big)\right],
$$

并满足 (P1) 中的候选、因果性和任务执行约束。该限制在提供可行资源分配的同时，将学习重点集中于计算感知的目的节点与路径选择。据此，本文采用以任务为条件的 GAT-PPO 方法，结合图编码、预测候选掩码和考虑预约的批次决策。图注意力网络与 PPO 分别承担状态表示和策略优化功能，资源规则与路由预测器则作为显式系统机制。[图注意力网络](https://arxiv.org/abs/1710.10903)、[近端策略优化](https://arxiv.org/abs/1707.06347)

---

## 写作说明与代码对应（不属于论文正文）

本中文版与 [英文版](SYSTEM_MODEL.md) 的小节、数学符号和建模假设对应，可作为中文论文正文初稿，也可用于逐段核对英文表述。正文按当前工程的实际执行语义撰写，没有引入第一研究点中的部分卸载、UAV 可控轨迹、能耗目标或任务优先级。

| 正文内容 | 当前工程对应 | 需要保留的建模边界 |
| --- | --- | --- |
| 圆轨道、ECI 坐标、动态选边 | `topology/walker.py`、`topology/graph_builder.py` | 理想化 Walker 场景，不称为真实 TLE/SGP4 轨道验证 |
| 批量任务、热点混合分布 | `tasks/task_generator.py` | 热点概率是混合权重，热点实际占比还含全网均匀抽样贡献 |
| 计算目标与完整路径候选 | `routing/candidate_builder.py`、`network/ksp.py` | 整任务卸载，参考大小用于路径排序，候选集有跳数和数量上限 |
| 存储转发、传播、CPU 服务、断链失败 | `env/event_engine.py` | 传播不占发送容量；数据完整到达后才能进入 CPU |
| CPU 与链路平方根分配 | `resource/cpu_allocator.py`、`resource/link_allocator.py` | 原始工作量决定权重；固定活动集合的静态代理解，不是动态全局最优 |
| CPU 剩余量与在途承诺量 | `compute/compute_queue.py`、`env/state_builder.py` | $Q/F$ 仅用于估计，不能叠加到仿真完成时延 |
| 未来发送区间检查 | `network/feasibility.py` | `H=0` 禁用未来检查；默认允许覆盖不足的候选，但不能称其已获保证 |
| 批次预约、增强代价、掩码与本地回退 | `agents/features.py`、`agents/ppo_agent.py` | 预约是策略内部代理量，不是执行器的物理预留；回退可能仍然超期 |
| 持有成本、一次性违约与失败惩罚 | `env/reward.py` | 默认违约后继续服务；奖励不是仅对已完成任务取平均时延 |
| 完成时延、成功率、截尾与固定窗口 | `env/metrics.py`、`evaluation/` | 完成时延是条件统计，必须同时报告失败和截尾情况 |

当前 `configs/base.yaml` 的主要设置为：24 颗卫星，$\delta t=0.25$ s，CPU 容量 20–100 Gcycles/s，ISL 容量 1 Gbit/s，平均每时隙 12 个任务，热点混合权重 0.3，$H_{\rm c}=3$，$H_{\rm p}=4$，$K=3$，$\eta_R=0.5$，$H=8$，$\beta=1$。这些是目前的实验配置，不能仅凭代码设定将其表述为已经过文献或实测校准的系统参数。$H=8$ 表示读取当前及之后 8 个快照；名义前视长度为 2 s，而完整缓存区间最多覆盖从当前边界起的 2.25 s。

当前惩罚系数为 $\alpha_{\rm d}=\alpha_{\rm f}=2$ s，固定归一化量 $Z=12$，PPO 默认 $\gamma=0.995$。任务持有成本包含所有实际执行时隙及排空时隙，评估指标则按指定到达窗口统计，两者不应混写成同一个指标。

若后续希望声称“直接优化平均完成时延”“严格满足期限”或“全局联合优化资源”，需要先改变目标或约束并同步修改实现。现版本更准确的描述是：在统一结构化资源分配下，学习目的计算节点与路由的联合选择，以降低折扣后的任务持有成本及违约、断链惩罚。该目标本身不保证提出的方法优于所有基线，性能结论仍需正式多随机种子实验支持。

上述三个外部链接分别用于计算感知路由背景、GAT 和 PPO 的来源说明。具体公式、假设、预测窗口和代价函数以本项目实现为准；论文正式排版时可将链接替换为 BibTeX 引用。
