# 强化学习实现与训练

当前动作是一个计算卫星，不包含路径。图路由在采样后执行，资源分配属于真实执行器，详见 [方法](METHOD.md)。训练配置为 `configs/ppo.yaml`，当前先使用 `configs/experiments/coupled24.yaml`；原 compute24/contact66 保留，六节点 `rl_smoke.yaml` 只做功能检查。

| 模块 | 职责 |
|---|---|
| `models/gat_encoder.py、models/actor_critic.py` | 边特征 GAT 或逐节点 MLP、图池化 |
| `models/destination_scorer.py` | 共享卫星节点评分，输出 S 个 logits |
| `models/actor_critic.py` | 节点分布与状态价值 |
| `models/rollout_tensors.py` | 每次 update 打包图和变长任务，批量重算冻结条件下的联合似然 |
| `agents/features.py` | 任务/图/节点状态，保存批次前缀和节点 mask |
| `agents/ppo_agent.py` | 节点采样、后置图路由、预约、PPO update、检查点 |
| `agents/rollout_buffer.py` | 物理时隙 transition 与 GAE |
| `agents/trainer.py、agents/episode_scenarios.py` | 全局 seed、不同回合场景、公平回放、验证选模、进度和保存 |

GAT 输入每节点 8 维、每有向边 4 维；全局上下文 12 维，任务 5 维，每节点额外决策特征 7 维。没有路径池化或联合动作特征，也没有手工完成时间 logit 先验。固定训练尺度保存在 checkpoint。节点换编号时共享评分等变，argmax 平局与图搜索平局仍可能依赖编号。

同批任务按期限排序逐个请求计算节点，然后求路径与更新预约；训练保存请求节点的 log probability，整个物理时隙的联合概率比用于 PPO clipping。路由拒绝与真实失败分开，拒绝惩罚用于让策略承担不可执行请求的代价。训练 rollout 为随机采样，验证与评估为确定性 argmax。

更新阶段一次打包 rollout，将 minibatch 的图、任务和节点评分合并为张量运算；逐任务 log probability/entropy 按原物理时隙归并。保留每个任务采样时的前缀状态、mask 和请求动作，不重算成独立动作，不在训练中重新路由。等价性检查覆盖 CPU/CUDA、GAT/MLP、变长任务、空时隙和梯度。在线采样仍逐任务执行预约，路由缓存只在同一 observation 内复用。

共享路由采用非负前缀代价下界提前终止无法改善已有最佳路径的搜索，保留同成本下的长度/字典序选择与搜索预算；该加速也作用于使用同一路由的基线。预约估计在日历提交前复用，不改变真实处理器共享规则。神经网络参数名、特征 schema=3 与奖励保持不变；数值求和顺序变化可能造成微小浮点差异，不能承诺旧训练轨迹逐位一致。

MLP 是逐节点共享编码加全局池化，同样使用任务/跳数/可达性特征、mask、路由、预约和 KKT。它属于编码消融，而不是纯 Vanilla PPO。当前没有为了加速降低 epochs、采样量或验证频率。

`coupled24` 设置 `rl.rollout_device=cpu`：小图的顺序决策在 CPU 上执行，批量反向仍在 `rl.device=auto/cuda` 的 GPU 上执行；每次更新完成和 checkpoint 加载后同步同一份权重，CPU 推理副本不独立学习。控制台 `[DEVICE SPLIT]` 与 manifest 分别记录采样和更新设备。通用默认 `same` 保留原设备行为。CPU/CUDA 的浮点计算与采样 RNG 流不同，因此改变采样设备需要作为新的运行配置记录，不能承诺旧轨迹逐位复现。

GAE、reward、预测价值和评估均在原单位；`value_scale=100` 只将 critic 输出及 value MSE 的数值尺度归一化。默认学习率 1e-4、gamma=.995、GAE=.95、clip=.2、entropy=.002、4 epochs、minibatch=64 个物理时隙、KL 上限 .03。critic 与策略共享编码器，不应将价值误差下降直接等同于性能收敛。

```bash
python -u scripts/train_ppo.py --config configs/experiments/coupled24.yaml \
  --set rl.updates=40 --set rl.device=auto --output results/new_gat
python -u scripts/train_ppo.py --config configs/experiments/coupled24.yaml \
  --set rl.encoder=mlp --set rl.updates=40 --output results/new_mlp
python -u scripts/evaluate_ppo.py --checkpoints results/new_gat/best.pt results/new_mlp/best.pt \
  --seeds 100 --allow-validation-reuse --algorithms local batch_greedy node_greedy \
  --output results/new_development
python scripts/plot_training.py results/new_gat
```

上例 coupled24 开启 `rl.randomize_episodes=true`。初始化及训练根 seed 固定为 2026，回合数据由根 seed、绝对回合编号和独立随机数域确定：重新生成任务和 CPU，保持热点数量/混合概率而改变热点卫星，并均匀选择一个轨道周期内的起始时刻。Walker 相位关系和物理参数保持不变；不根据连通性或方法表现筛掉回合。GAT、MLP 的同编号回合具有相同任务/CPU/拓扑指纹；模型及动作采样 RNG 不会消耗场景随机数。

`episode_scenarios/episode_*.json` 保存完整回合配置、CPU、根 seed、派生回合 seed、轨道偏移、热点及输入指纹，可重建任务与轨道。checkpoint 保存绝对 `episode_count` 和场景协议版本，恢复按该编号继续生成；无需依赖进程中尚未保存的随机数状态。验证 100 和参考历元保持固定，开发评估复用 100。独立测试需改用未参与训练或选模的 seed 并去掉 `--allow-validation-reuse`；目前先做开发比较。主实验批量入口见 [SERVER_EXPERIMENTS.md](SERVER_EXPERIMENTS.md)。

通用默认 `randomize_episodes=false` 仅用于读取旧固定回放协议；旧场景训练结果不因本次修改而改变。新 coupled24 协议需使用全新目录，不从旧 fixed_replay checkpoint 续训。

`best.pt` 按验证平均 reward 选取，`last.pt` 保存最后状态。恢复保存模型、Adam、采样随机状态与 update 日程；写入新目录，从同一新版配置的 last.pt 增加总 updates。特征 schema=3 与旧 schema=1/2 不兼容；不要恢复旧服务器 20 updates 权重。

控制台自动保存完整文本，显示实际 CPU/GPU、FPS、主要任务指标、PPO 数值指标与输出路径。`routing_curve.png` 显示拒绝、屏蔽概率、实际断链与期限违约。短功能检查只确认可运行，不能证明提出的方法优于 node_greedy。
