# 2026-10-10 表示层优化与并行实验

分支：`exp/20261010-01-kkt-task-gated-graph-ppo`。本轮保持 coupled24 的轨道、链路、到达率、任务分布、奖励、资源分配和 PPO 超参数，检验竞争状态与图表示是否能改善计算星选择。动作仍是一个计算卫星，接触路由在选择之后执行。

## 已实现的方法与对照

| variant | 编码器 | 特征 | 邻居 | 用途 |
|---|---|---|---|---|
| full | GAT | base | 物理图 | 历史 GAT 对照 |
| mlp | 共享 MLP | base | 不聚合 | 历史 MLP 对照 |
| self_graph | GAT | base | 仅自环 | 分辨邻居传播与深度/归一化的作用 |
| mlp_kkt | 共享 MLP | kkt | 不聚合 | 竞争特征本身的作用 |
| gat_kkt | GAT | kkt | 物理图 | 同样竞争特征下的图编码作用 |
| gated_kkt | 自身 MLP＋任务门控 GAT | kkt | 物理图 | 保留自身状态，按任务引入图信息 |

MLP 对照同样使用接触路由、mask、预约和 KKT；它不是裸 Vanilla PPO。默认 tmux 启动后四项，各自从零训练 200 updates、400 episodes。历史 full/mlp 的 200-update 结果只在输入指纹一致时作为参考合并，不把新模型暖启动后的步数与从零训练混为一谈。也可显式启动全部六项。

## KKT 竞争状态

默认节点/目的星特征维度仍为 8/7，checkpoint schema=3。`rl.feature_set=kkt` 的维度为 11/12，schema=4。

节点新增三个已知状态：活跃 CPU 任务数、CPU 驻留任务的剩余 cycles 平方根之和、在途任务的剩余 cycles 平方根之和。目的星决策新增活跃数、CPU 平方根总量、在途平方根总量、同批此前实际提交节点上的预约平方根总量，以及共享服务代理。

对当前任务计算量 $L_u$，目的星容量 $F_s$，当前 CPU 集合 $\mathcal J_s$ 和同批已预约集合 $\mathcal B_s$，代理为

$$
\widehat T_{u,s}^{share}=\frac{L_u}{F_s}\left(1+\frac{\sum_{j\in\mathcal J_s}\sqrt{L_j^{rem}}+\sum_{j\in\mathcal B_s}\sqrt{L_j}}{\sqrt{L_u}}\right).
$$

无预约、立即加入当前 CPU 集合时，它等于当前 KKT 分配下的 $L_u/f_u$。由于服务份额会随事件变化，远程任务尚未到达 CPU，同批预约也不是已执行资源，所以它**不是实际计算完成时间，不是 FIFO 等待时延，也不是物理可行性下界**。它作为神经网络输入，不用于增加 deadline 排除条件。在途工作量分开提供，避免当成已经开始 CPU 竞争的任务。equal 分配对照按当前和已预约任务数构造代理。

总量按固定训练尺度归一化，平方根总量除以 $\sqrt{F_sT_0}$，再用 `signed_log` 压缩。新增信息只来自当前观察和已采样批次前缀，不能读取未来任务。

## 任务门控双分支

自身分支为两层共享 MLP，图分支为已有两层边特征 GAT。图分支在每个物理时隙编码一次；每项任务在评分时计算

$$
\mathbf h_{u,s}=\mathbf h_s^{self}+\mathbf g_{u,s}\odot\mathbf h_s^{graph}.
$$

门控输入包含自身目的星、源星、全局上下文、任务编码和目的星决策特征。门控是 sigmoid，末层权重初始为零、bias=-3，初始图修正较小但可训练。价值网络使用自身分支的池化表示，因此图分支仅收到策略与熵目标的梯度。自身分支仍由 actor/critic 共享。

当前 GAT 已有残差连接；新增独立自身分支用于明确保留节点状态，不是简单再添加一条残差。没有添加手工时延 logits，也没有固定卫星 ID 输出层。边特征仍参与原 GAT 注意力权重，本轮未同时引入新的边消息网络，避免把多项结构变化混成单个对照。

## PPO、兼容性与诊断

保存每次采样时的 KKT 特征、批次前缀和 mask；更新时重算这些冻结条件下的概率。仍按整个物理时隙求联合动作概率比，不改为独立任务 PPO。CUDA 更新与 CPU rollout 的参数在每次 update 后同步。

历史 schema=3 节点策略可加载；schema=4 根据配置构建正确输入维度，禁止将其直接用于旧特征。`--resume` 仍只允许修改设备与目标 updates，不允许用旧模型静默续训新架构。新方法本轮均从零开始。

日志及 updates.csv 新增图编码器、自身编码器、actor 评分头、critic 价值头的裁剪前梯度范数及图门控均值。门控均值针对更新 minibatch 中任务的全部节点与 hidden 通道，不是所选节点的概率。这些是模块级范数，不能独自证明共享层梯度冲突。manifest 保存参数量、特征维度、邻居模式、critic 分支及 gate 初始化。

## /tmp 与 tmux 启动

在服务器独立分支检出中的 `leo_compute_routing` 目录执行：

```bash
PY=/home/zhaojunan_25/.conda/envs/leo-contact/bin/python
RUN=/tmp/zhaojunan_leo_experiments/20261010_01_kkt_gated/runs

"$PY" scripts/launch_representation_tmux.py \
  --run-root "$RUN" --session leo-20261010-01 --updates 200

tmux attach -t leo-20261010-01
```

一个 monitor 窗口加四个训练窗口，用 `Ctrl+b` 再按 `w` 切换窗口；`Ctrl+b` 再按 `d` 退出查看并保留任务运行。每项方法的结果在 `$RUN/results/<variant>/`，全量控制台在 `$RUN/console/<variant>.log`，状态和进程号在 `$RUN/status/<variant>.json`。

```bash
"$PY" scripts/representation_status.py "$RUN"
tail -f "$RUN/console/gated_kkt.log"
```

启动器需要 CUDA、tmux 和至少 10 GiB 空间，检查输出目录及 session 未被占用。各窗口使用同一环境、独立进程，设置单线程数学库，把 `TMPDIR`、Python 字节码、Matplotlib、Torch 和 CUDA 缓存放到 `$RUN` 下。不向满载的 home 写新训练数据，也不重置已有实验。

完成后自动汇总 `comparison.csv/json`，先检查任务、CPU、拓扑和 seed 指纹一致。可以用 `--reference-results /tmp/.../旧结果根目录` 加入历史 full/mlp。评估继续使用 seed=100，并明确标记为验证复用；不是独立测试，不生成单 seed 的置信区间结论。所有方法共享根 seed=2026、派生回合序列和评估输入。

结果判据：同时比较平均时延、P95、成功率、路由回退、CPU 阶段耗时、热点分配负载、FPS 和决策耗时。若新图模型只优于贪心而不优于同特征 MLP，不能据此声称图编码有收益。实验完成后下载保留日志、指标及最佳权重。
