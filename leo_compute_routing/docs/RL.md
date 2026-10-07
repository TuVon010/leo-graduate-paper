# 强化学习实现与训练

当前动作是一个计算卫星，不包含路径。图路由在采样后执行，资源分配属于真实执行器，详见 [方法](METHOD.md)。训练配置为 `configs/ppo.yaml`，物理场景为 `configs/experiments/compute24.yaml`、`contact66.yaml`；六节点 `rl_smoke.yaml` 只做功能检查。

| 模块 | 职责 |
|---|---|
| `models/gat_encoder.py、models/actor_critic.py` | 边特征 GAT 或逐节点 MLP、图池化 |
| `models/destination_scorer.py` | 共享卫星节点评分，输出 S 个 logits |
| `models/actor_critic.py` | 节点分布与状态价值 |
| `agents/features.py` | 任务/图/节点状态，保存批次前缀和节点 mask |
| `agents/ppo_agent.py` | 节点采样、后置图路由、预约、PPO update、检查点 |
| `agents/rollout_buffer.py` | 物理时隙 transition 与 GAE |
| `agents/trainer.py` | 回放、固定 seed、验证选模、进度和保存 |

GAT 输入每节点 8 维、每有向边 4 维；全局上下文 12 维，任务 5 维，每节点额外决策特征 7 维。没有路径池化或联合动作特征，也没有手工完成时间 logit 先验。固定训练尺度保存在 checkpoint。节点换编号时共享评分等变，argmax 平局与图搜索平局仍可能依赖编号。

同批任务按期限排序逐个请求计算节点，然后求路径与更新预约；训练保存请求节点的 log probability，整个物理时隙的联合概率比用于 PPO clipping。路由拒绝与真实失败分开，拒绝惩罚用于让策略承担不可执行请求的代价。训练 rollout 为随机采样，验证与评估为确定性 argmax。

GAE、reward、预测价值和评估均在原单位；`value_scale=100` 只将 critic 输出及 value MSE 的数值尺度归一化。默认学习率 1e-4、gamma=.995、GAE=.95、clip=.2、entropy=.002、4 epochs、minibatch=64 个物理时隙、KL 上限 .03。critic 与策略共享编码器，不应将价值误差下降直接等同于性能收敛。

```bash
python -u scripts/train_ppo.py --config configs/experiments/compute24.yaml \
  --set rl.updates=40 --set rl.device=auto --output results/new_gat
python -u scripts/train_ppo.py --config configs/experiments/compute24.yaml \
  --set rl.encoder=mlp --set rl.updates=40 --output results/new_mlp
python -u scripts/evaluate_ppo.py --checkpoints results/new_gat/best.pt results/new_mlp/best.pt \
  --seeds 100 --allow-validation-reuse --algorithms local batch_greedy node_greedy \
  --output results/new_development
python scripts/plot_training.py results/new_gat
```

上例固定一个训练回放 seed 2026，验证 100，开发评估复用 100。独立测试需改用未参与训练或选模的 seed 并去掉 `--allow-validation-reuse`；目前先做开发比较。主实验批量入口见 [SERVER_EXPERIMENTS.md](SERVER_EXPERIMENTS.md)。

`best.pt` 按验证平均 reward 选取，`last.pt` 保存最后状态。恢复保存模型、Adam、采样随机状态与 update 日程；写入新目录，从同一新版配置的 last.pt 增加总 updates。特征 schema=3 与旧 schema=1/2 不兼容；不要恢复旧服务器 20 updates 权重。

控制台自动保存完整文本，显示实际 CPU/GPU、FPS、主要任务指标、PPO 数值指标与输出路径。`routing_curve.png` 显示拒绝、屏蔽概率、实际断链与期限违约。短功能检查只确认可运行，不能证明提出的方法优于 node_greedy。
