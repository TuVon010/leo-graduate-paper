# 新版实现验证记录

## 2026-10-09：服务器实测耗时与预实验检查

同一 compute24、GAT、2 episodes、4 epochs、304 次优化步骤的首个 update：旧实现采样 97.99 s、优化 144.26 s；新版采样 104.00 s、优化 3.58 s。排除验证后合计由 242.25 s 降至 107.58 s，约 2.25 倍；优化部分约 40 倍。新版首 update 额外执行了最终验证，因此不把包含不同验证日程的总耗时直接比较。服务器冻结 minibatch 微基准的约 141/193 倍仅指 GAT/MLP 评分与反向，不是整套训练速度。

coupled24 的 CPU 采样 / CUDA 更新完整首 update：GAT 总耗时约 215 s，MLP 约 212 s，各包含两条 600 s 训练回放与一次最终验证；优化分别约 3.91/3.11 s。该场景比旧 compute24 更拥塞，不能据此直接比较采样设备的速度。40 updates、每 5 updates 验证的 main 两种编码器顺序运行，按冷启动耗时估计约 3–3.5 h，加上基线评估略有增加；策略改善后排空与队列负担可能下降，实际时间以日志 ETA 为准。

另用完全相同的初始观测、权重及 5 个任务测量确定性节点选择与路由，CPU/CUDA 动作一致；预热后 30 次中位数分别为 4.69/9.99 ms。该小图决策检查支持当前 CPU 采样配置，不代表整个 episode 恒定快 2.13 倍。原始数据为 `results/rollout_device_speed_20261009.json`。

seed 100 的 coupled24 预检查：computing_aware 成功率 93.23%、任务代价 2.144 s，node_greedy 成功率 95.35%、任务代价 1.910 s，两者截尾均为 0。冷启动 1 update 的 GAT/MLP 验证成功率仅约 49.48%/50.38%，队列与远程拒绝偏高，不构成方法效果验证；现阶段只确认实现可运行和训练加速。

服务器保留路径：`results/optimized_compute24_timing_20261008/`、`results/coupled24_full_timing_20261008/`、`results/coupled24_preflight_100m_20261009/`，以及 150 Mbit/s 的参数控制 `results/coupled24_preflight_150m_20261009/`。原 66 星进程在保留 update 7 的 last.pt 后停止；历史 24/66 星数据没有删除。

## 2026-10-08：24 星场景与训练加速

- 当前优先 `coupled24`，服务器入口默认只运行该 24 星场景。原 compute24、66 星配置与全部历史结果保留。
- 本地与 Ubuntu/RTX 4070 Ti 服务器均通过 **123 项测试**，包含 CPU/CUDA、GAT/MLP 批量似然、熵、价值与梯度等价性、变长任务/空时隙、CPU 采样副本权重同步和 checkpoint 加载。
- 带预测预约的小图路由结果与穷举代价比较一致；缓存估计提交不重复计算。共享路由的加速也作用于基线。
- schema=3 的网络参数结构保持兼容；coupled24 改变物理配置，必须新建训练目录，不续用旧场景的训练轨迹。
- 70°/100 Mbit/s/20 任务每秒/8 s 前视场景经过轨道、负载、期限下界和空系统路由暴露审计；详见 [SCENARIOS.md](SCENARIOS.md)。
- `results/batched_speed_check_20261008.json`、`results/batched_speed_cpu_check_20261008.json` 为本地冻结 rollout 微基准，不属于模型效果实验。服务器完整时域测速保存在 `results/optimized_compute24_timing_20261008/`、`results/coupled24_full_timing_20261008/`。
- 新实现没有改变采样量、4 epochs、PPO 联合概率目标、GAE、奖励或资源分配。CPU 采样仍为同权重模型，GPU 用于批量更新；微基准加速倍数不能当作整套训练加速倍数。

## 2026-10-07 记录

日期：2026-10-07。当前架构为计算卫星 GAT-PPO → 后置预测接触图路由 → KKT 资源共享，feature schema=3。旧联合动作、KSP、完成时间 logit 先验及旧配置已移除；既有 `results/` 数据和日志保留，历史说明见 [HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md)。

## 回归与接口

- `python -m pytest -q`：**114 passed**。覆盖原物理守恒/事件边界、双向共享、接触半开窗口、预测覆盖隔离、动态 CPU 服务、GAE 与概率快照、实际梯度、恢复与设备迁移。
- 新版测试验证节点动作数量等于卫星数、节点重编号等变、只保存请求节点概率、后置路由与回退记录、预约不改变执行器、拒绝惩罚与实际断链分开统计。
- 验证相同固定权重与特征尺度支持 24/48/72/96 星；旧 schema 权重和旧路由配置参数明确拒绝加载。
- 路由小图与穷举路径代价一致；扩展预算截断明确记录。加入到目标的最短跳数下界剪枝和带预约的路径前缀积分复用，验证与完整积分一致。

## 实际运行

| 检查 | 保存位置 | 说明 |
|---|---|---|
| GAT CPU 2 updates | `results/hierarchical_smoke_20261007/` | 六节点功能检查，保存训练/验证及 best/last |
| 服务器 main 全流程 | `results/hierarchical_runner_check_20261007/` | GAT、MLP 各 2 updates，单 seed 100 开发评估与基线，绘图、日志、角色映射和完成标记 |
| 最终评估配置 | `results/hierarchical_evaluation_finalcheck_20261007/` | 单 seed 100、GAT/MLP 与 7 基线，移除重复规则命名 |
| compute24 CUDA | `results/hierarchical_compute24_check_20261007/` | 16 到达时隙、1 update，RTX 4060 实际 GPU 前向/反向与保存 |
| contact66 CPU | `results/hierarchical_contact66_finalcheck_20261007/` | 16 到达时隙、1 update；验证汇总正确打印路由拒绝率 |
| 接触路由审计 | `results/hierarchical_audit_check_20261007/` | 16 时隙、4 个均匀抽样任务、seed 100；新逐目标审计 CSV 与校准 |

上述轨道短检查保持对应物理场景，只缩短时域和训练量，不作为正式性能实验。两个轨道检查均完成全部任务、无截尾和数值异常。短训练尚未收敛，尤其初始策略存在较多路由拒绝；不能据此认定 PPO 已优于贪心。

服务器 main/routing/modules/resource/scale/all/audit/sensitivity 共 8 组命令计划完成 dry-run 校验；剩余 YAML 配置及继承通过校验。单 seed 开发比较不生成置信区间，复用验证的角色在 study manifest 和控制台中明确标注。

## 图与文档

中英文系统模型、方法、工程结构、物理参数和服务器命令均同步至三层流程。PNG/SVG/PDF 系统图已重新生成，并将两份 PDF 用 Poppler 渲染后逐一检查，未发现裁切、重叠或乱码。数学式中的排队、在途与预测预约均对应实现，未将 Q/F 叠加到实际时延，也未声称参考预测保证成功。

现有 Python 3.11 / PyTorch 2.7.1+cu126 环境继续使用，此次未安装新包。控制台原文在相应 `console_logs/`，批量子日志在输出根目录 `logs/`。正式服务器训练应使用全新目录重新开始，不能恢复旧 20 updates 权重。
