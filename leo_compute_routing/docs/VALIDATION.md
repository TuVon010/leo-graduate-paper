# 第一阶段验证记录

## 服务器单 seed 入口：2026-10-05

- 完整回归 **99 passed**；新增单 seed 评估/冻结泛化检查，验证 aggregate 与 cost_shift 的 CI 为空，manifest 明确标记 pilot。
- 新入口 `scripts/run_server_experiments.py` 默认训练初始化 2026、测试 201，覆盖 main / shield / modules / resource / scale / audit / sensitivity，支持 dry-run 与新目录 last.pt 续训。
- 合成 CPU 检查完成 GAT、MLP 各 1 update，单测试 seed=201 与 local 对比，生成训练图与角色映射；继续至 2 updates，验证恢复训练与独立新目录评估。
- resource 检查额外训练 equal，并分别使用 full/sqrt 与 equal 的检查点配置评估，没有将 equal 模型切换为 sqrt；已完成模型可复用。
- all、scale 与 15 点 sensitivity 的命令计划检查通过。长轨道正式训练没有在本次本地执行，不能据此声称收敛或方法优势。
- 合成结果：`results/server_entry_check_20261005/`、`results/server_entry_resume_check_20261005/`；最终入口复核另存 `results/server_entry_final_check_20261005/`。Linux 安装和命令见 [SERVER_EXPERIMENTS.md](SERVER_EXPERIMENTS.md)。

## 接触窗口方法升级：2026-10-05

- 最终完整回归：**98 passed**，包含原 72 项及新增接触/预约/泛化边界测试。

- 新增 contact 候选搜索、有限窗口容量积分、无向链路/CPU 时间预约和预测 Shield；特征 schema=2。
- 边界测试覆盖半开接触、变速发送、传播后无需链路、窗口外信息隔离、双向批次超订、CPU 输入到达约束及剩余工作不重复计入。
- 同构节点/候选重编号下 MLP/GAT 全评分头等变；同一固定权重及归一化可执行 24/48/72/96 星，完整泛化脚本验证 held-out 参照、检查点不变和结果保存。
- CUDA 的 `contact_smoke.yaml` 已完成 2 updates，并在 seed=201、202 与既有基线完成统一轨迹评估；这是合成检查，不是轨道证据。
- 新方法 24 星 Walker 短轨道训练已完成 2 updates：到达阶段 24×0.25=6 s、最大排空 20 s；训练 seed=0、1、验证 seed=100。保存检查点、日志及学习/Shield 曲线。训练期限不合规与实际违约统计单独报告。
- 48 星 CUDA 短训练完成 1 update；同一个检查点经实际 CLI 在 24/48/72/96 星、seed=201/202 完成冻结评估，见 `zero_shot_short/`。各规模均为 6 s 到达窗口；48 星 held-out 成功率约 51%–54%，其他规模约 52%–72%，没有收敛或性能优势含义。
- 四组正式规模配置在 600 s 内每 10 s 抽查几何，61 个样本均连通，记录为 `scale_geometry_samples.json`；这是稀疏几何抽查，不能替代完整时隙审计。
- 原始记录位于 `results/contact_upgrade_20261005/`。旧 schema-1 检查点不在新网络自动迁移，正式 600 s 场景仍需重新训练。

完整接触窗口功能检查不等同于已实现等待式 CGR、正式安全保证或算法收敛；参见 [CONTACT_METHOD.md](CONTACT_METHOD.md)。

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
