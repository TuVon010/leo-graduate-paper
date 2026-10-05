# 第一阶段验证记录

## 强化学习阶段更新：2026-10-05

- 按用户授权，在项目 `.conda-env` 安装 `torch==2.7.1+cu126`；Python 3.11.17。官方 wheel SHA256 为 `f3af23387ac106b5b01dbef0eb021883e0c00ff4073477b7ce1cbade5ef5038d`，已验证一致。其他 Conda 环境没有修改。
- CUDA runtime 12.6，RTX 4060 Laptop GPU；CUDA 可用，GPU 矩阵前向/反向、NumPy 2.4.6 互操作通过；`pip check` 无冲突。
- **72 passed**，包括原有仿真测试、GAE episode 边界、联合 log probability 重评分、mask 快照、全不可行 fallback、GAT 节点重标号等变性与非邻居隔离、真实梯度更新、检查点、CPU/CUDA 加载、恢复训练与独立 seed 保护。
- `configs/rl_smoke.yaml`：MLP-PPO 在 CPU、GAT-PPO 在 CUDA 各完成 2 updates，保存 `best.pt`、`last.pt`、周期检查点、训练/验证日志。
- 两种模型与原有五种基线在 held-out seed=201、202 完成共同 trace 评估；原始结果、指纹和 checkpoint SHA256 已保存；学习曲线已生成并检查。
- 项目环境的结果目录为 `results/rl_verification_20261005/`，训练说明见 [RL.md](RL.md)，完整依赖快照为 `requirements-rl-lock.txt`。

这些是短合成场景的功能验证。尚未执行默认 200-update 轨道正式训练，不说明模型已收敛或优于基线。Oracle 与正式论文实验仍未完成。

## 原仿真阶段记录

日期：2026-10-04。本记录用于确认实现可运行，不是多 seed 论文结论。

**状态更新：下文为旧配置的历史运行记录。此前列出的 `results/verification_*` 原始目录已按清理请求删除，路径目前不能作为现存数据证据。旧默认参数现保存为 `configs/legacy_overload.yaml`。**

本次修订在现有 `.conda-env`（Python 3.11.17、NumPy 2.4.6、NetworkX 3.6.1、PyYAML 6.0.3、pytest 9.1.1）中通过 **55 项测试**。五 seed 修订 Walker 预实验与参数审计已重新保留在 `results/review_20261004/`；指标、CI 与限制见 [项目复核](PROJECT_REVIEW.md)。没有安装或修改环境包。

## 已运行的检查

- `python -m pytest -q`：**44 passed**。覆盖闭式解 KKT 与容量约束、跨时隙竞争、双向共享链路、计算工作量守恒、逐跳存储转发、数据到达后 CPU 服务、deadline 与断链边界、删失、候选 mask、配置与回放。
- 6 节点 smoke：五种基线均跑完，无删失；见 `results/verification_smoke/`。
- 24 节点圆轨道 Walker：五种基线均跑完，处理同一批 **2801** 个任务，无删失；见 `results/verification_walker_v2/`。
- 合成接触压力测试：285 个任务；可观测到链路中断；见 `results/verification_contacts/`。
- 消融脚本：future、等分资源、deadline mask，均能独立运行；移除 future 保留 deadline mask；见 `results/verification_ablation/`。
- 参数扫描：两个到达率、两个 seed，统一回放各策略；见 `results/verification_sensitivity/`。
- 缓存与任务回放：使用压力测试保存的 YAML、NPZ、JSON 重新执行，除三个耗时字段外，全部汇总指标相同；五个算法的 `tasks.csv` 和 `slots.csv` 均逐字节一致；见 `results/verification_replay/`。
- 绘图脚本运行成功，已检查图像标签与布局；见压力测试目录 `summary.png`。

运行结果目录是本次本机生成的数据，已列入 `.gitignore`；源代码和配置可以重新生成它们。

## 默认 Walker 结果

seed=42；24 颗卫星；产生任务窗口 60 s；默认超过 deadline 继续执行。

| 算法 | 已完成任务平均时延 / s | 按时成功率 | 路由失败率 | 删失率 |
|---|---:|---:|---:|---:|
| local | 108.9398 | 0.2292 | 0 | 0 |
| shortest_offload | 136.5937 | 0.1639 | 0 | 0 |
| least_load | 1.5343 | 0.4777 | 0 | 0 |
| computing_aware | 1.7477 | 0.3788 | 0 | 0 |
| computing_aware_future | 1.7365 | 0.3731 | 0 | 0 |

该配置中网络快、热点计算拥塞明显，未来拓扑筛选没有改善成功率，计算感知启发式也没有超过最小负载。不能从“代码跑通”推导 proposed 算法必然更优。

一个后续需要检验的解释是：当前启发式对批次任务独立评分，同批次集中选中较强 CPU 时会产生共同资源竞争；估计的独占执行时间并未反映这部分竞争。可增加有虚拟预约的批次贪心基线，再判断 PPO 是否还有优势。这是待检验的原因，不是已证实的归因。

## 合成压力测试结果

seed=42；8 节点合成周期图；开启 deadline 丢弃。这组结果仅验证逻辑，不代表真实轨道收益。

| 算法 | 按时成功率 | 路由失败率 |
|---|---:|---:|
| local | 0.7544 | 0 |
| shortest_offload | 0.2281 | 0.0456 |
| least_load | 0.8000 | 0.1333 |
| computing_aware | 0.9298 | 0.0526 |
| computing_aware_future | 0.9754 | 0 |

严格消融中仅关闭 future、保留 deadline 筛选时，成功率回到 0.9298、路由失败率 0.0526。换成等分资源时，同样获得 0.9754 的成功率，而已完成平均时延从约 0.9743 s 降到约 0.9302 s，进一步说明平方根分配不保证动态指标始终优于等分。

后续论文实验应在真实轨道场景使用多个 seed、统一评估窗口和经依据校准的输入数据与带宽，报告置信区间及预测检查开销。
