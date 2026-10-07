# 工程接口与建模约定

本项目采用图强化学习卸载 → 预测图路由 → KKT 动态资源共享。详细公式见 [中文系统模型](SYSTEM_MODEL_ZH.md)，算法说明见 [METHOD.md](METHOD.md)。

`LeoEnv.reset()` 返回当前图、任务批次、节点/边状态、活动任务快照和有限未来 `ContactPlan`。观测中没有路径动作列表。策略只读取观测并返回 `{task_id: RoutingAction}`；远程裸整数不能提交执行器，需先经图路由解析。兼容本地 `task_id: source_sat` 的简写。

```python
obs, info = env.reset()
while True:
    actions = policy.select(obs)
    obs, reward, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        break
```

学习策略内部先采样计算卫星，随后调用 `ContactAwareRouter.resolve()`。`RoutingAction` 是实际执行指令，而非 PPO 联合动作；字段为任务 ID、实际计算卫星、路径、可选原请求节点、回退原因。训练 buffer 保存请求节点及决策时刻的冻结特征和 mask，不能用回退节点重算 PPO 概率。

`routing/reservations.py` 只预测预约时段，不锁定实际资源。`env/event_engine.py` 以事件重分配链路/CPU，整输入逐跳发送、两个方向共享 ISL、数据到达后再计算。CPU 使用处理器共享，Q/F 不能再次加入实际时延。失败输入不预约下游 CPU；未完成任务跨时隙保留工作量。

路由搜索只针对已选卫星，使用当前图无环路径与有限前视。`path_expansion_limit` 是内部计算预算，截断单独记录。当前不等待断开链路重连，不在执行中重路由。

所有物理配置使用 SI：bit、cycles、cycles/s、bit/s、m、s。`topology/` 决定轨道与 ISL，强化学习不修改物理拓扑。`resource/` 的平方根规则为静态代理 KKT 解，运行时不断重分配。`evaluation/` 使用共同任务/CPU/拓扑指纹与统一统计窗口。

扩展新图编码器时保持卫星共享评分和变长节点输出；扩展路由时保持计算目的节点由 PPO 指定；扩展资源规则时同步修改系统模型和资源消融。旧联合节点—路径实现与参数已移除，旧检查点需重新训练。
