# 通信—计算耦合与轨道接触实验场景

**方法版本更新：** 本文主要描述物理场景与升级前预检查。新推荐方法通过 `configs/contact_ppo.yaml` 继承这些物理参数，启用 contact 候选与预测预约 Shield；跨规模及模块消融的最新入口见 [CONTACT_METHOD.md](CONTACT_METHOD.md)。原五组配置保留用于控制对照，历史预检查不代表新算法的性能结果。

可以在有依据的参数范围内，构建让计算负载、链路竞争和接触变化都进入决策的场景。本次增加一组用于检验这些机制的配置，并保留原 `base.yaml`。配置依据工作量与容量比例、几何连通性和发送时间尺度确定，没有用 GAT/PPO 的最终排名来挑选参数。

新场景仍采用圆轨道二体 Walker-Delta、地球遮挡检查、节点度数限制及真实轨道角速度。任务执行器、资源分配器、基线和 reward 的形式均保持一致。论文性能结论需要在这些配置上重新训练和测试。

## 1. 五个场景的用途与参数

| 参数 | 匹配低负载对照 | 通信—计算耦合主场景 | 计算密集 | 链路受限 | 高倾角接触场景 |
| --- | --- | --- | --- | --- | --- |
| 配置 | `walker_low_load_control.yaml` | `walker_coupled.yaml` | `walker_compute_heavy.yaml` | `walker_link_heavy.yaml` | `walker_contact_dynamic.yaml` |
| 星座 | 3 面 × 8 星 | 3 面 × 8 星 | 同主场景 | 同主场景 | 6 面 × 11 星 |
| 高度 / 倾角 | 600 km / 53° | 600 km / 53° | 同主场景 | 同主场景 | 780 km / 80° |
| 最大链路距离 | 6000 km | 6000 km | 同主场景 | 同主场景 | 6400 km |
| 有效 CPU 容量 | 40–100 Gcycles/s | 40–100 Gcycles/s | 同主场景 | 同主场景 | 同主场景 |
| 任务业务 ISL 容量 | 200 Mbit/s | 200 Mbit/s | 200 Mbit/s | 100 Mbit/s | 100 Mbit/s |
| 输入数据量 | 20–60 Mbit | 20–60 Mbit | 同主场景 | 同主场景 | 同主场景 |
| cycles/bit | 500–1500 | 500–1500 | 1000–2000 | 500–1500 | 500–1500 |
| 每时隙期望任务数 | 1 | 4 | 4 | 4 | 11 |
| 期限 | 2–8 s | 2–8 s | 同主场景 | 同主场景 | 同主场景 |
| 热点混合权重 / 热点数 | 0.2 / 3 | 0.2 / 3 | 同主场景 | 同主场景 | 0.2 / 8 |
| 时隙 | 0.25 s | 0.25 s | 同主场景 | 同主场景 | 同主场景 |
| 到达阶段 / 最大排空阶段 | 600 s / 300 s | 同左 | 同左 | 同左 | 同左 |
| 名义预测长度 | 4 s（H=16） | 同左 | 同左 | 同左 | 同左 |

文件均位于 `configs/experiments/`，继承 `ppo.yaml`，可以直接用于 RL 训练。原有 `base.yaml`、`medium.yaml`、`large.yaml` 仍可作为已有配置运行。

低负载对照只降低主场景的到达率。计算密集场景只改变 cycles/bit；链路受限场景只改变通信服务容量。接触场景同时改变星座规模、几何和总到达率，属于另一类轨道场景，不应把其与主场景的差异全部归因于倾角。各方法的 future 消融仍须在同一配置内完成。

## 2. 参数依据与建模边界

### 2.1 任务变大，通信和计算都需要时间

20–60 Mbit 等于 2.5–7.5 MB，可以解释为一次待处理的图像块或聚合数据块。它是实验任务粒度假设，没有绑定具体传感器。

主场景的平均输入量与平均计算工作量为

$$
\mathbb E[D]=40\times10^6\ \mathrm{bit},\qquad
\mathbb E[W]=\mathbb E[D]\mathbb E[C]=40\times10^9\ \mathrm{cycles}.
$$

取 CPU 平均容量 70 Gcycles/s，独占计算时间量级约为 0.57 s。主场景单跳满容量发送时间约为 0.20 s，按参考速率系数 0.5 估计则约为 0.40 s；链路受限场景对应约为 0.40 s 和 0.80 s。多跳、资源共享和在途计算需求会进一步改变这些时间。

因此，该场景使远程计算节省和路径通信代价处于能够相互影响的量级。以上是无竞争时间代理，不是仿真实际时延，也不说明任一算法必然更好。

### 2.2 CPU 能力与任务需求使用相同的周期单位

40–100 Gcycles/s 位于原方案 20–100 Gcycles/s 范围内，表示留给所研究任务的有效处理能力，不是物理 CPU 主频。cycles/bit 仍需由任务 profiling 或明确的业务假设支持。

[Computing-Aware Routing for LEO Satellite Networks](https://arxiv.org/html/2211.08820v1) 采用 GFLOPS/GFLO 表示计算能力和需求，并扫描通信能力与任务大小。它支持研究通信—计算比例的思路，但不能直接把 GFLOPS 等同于本项目的 cycles/s。本文保留这一区别。

### 2.3 100–200 Mbit/s 是任务业务可用容量

NASA 的通信技术报告介绍了 [CubeISL 的 100 Mbit/s、约 1000 km 跨星链路设计](https://www.nasa.gov/smallsat-institute/sst-soa/soa-communications/)；[CLICK B/C 的目标](https://www.nasa.gov/smallspacecraft/what-is-click/) 则包括超过 20 Mbit/s、25–580 km 的跨星通信。这说明较低服务速率有研究背景，但这些终端的距离范围不能直接用于本项目的数千公里链路。

本组配置将 100–200 Mbit/s 解释为研究任务业务获得的有效服务预算。上述计算感知路由论文还介绍了长距、多 Gbit/s ISL，允许将更低的任务业务预算理解为共享平台上分配的一部分能力。具体预留比例属于工程假设；本项目没有模拟其他业务的到达、终端捕获过程或完整光学链路预算，因此不能声称复现了某款终端。

### 2.4 局部竞争与全网余量同时存在

主场景每秒期望到达 16 个任务。使用平均 CPU 容量估计，全网计算负载为

$$
\rho_{\rm global}\approx
\frac{16\times40\times10^9}{24\times70\times10^9}
\approx0.381.
$$

热点源概率是混合分布：单热点概率为 $0.2/3+0.8/24=0.1$，三个热点合计产生约 30% 任务，不能写成 20%。单热点的平均到达工作量为 64 Gcycles/s；热点 CPU 较弱时可能出现局部积压，但网络整体还有计算余量。

计算密集场景的平均工作量增加到 60 Gcycles，平均全网负载约为 0.571。该设计检验通过跨星选择分散工作量的能力，而不是让整个系统必然处于持续过载。实际每个 seed 的负载以审计文件为准。local-only 可能持续积压，必须同时报告违约与截尾情况。

### 2.5 期限与预测窗口匹配任务时间尺度

期限为 2–8 s。主场景最大任务工作量为 90 Gcycles；计算密集场景为 120 Gcycles。正式运行前应检查每个 seed 的最快 CPU 独占下界是否超过期限，避免把先天不可行任务与调度失败混为一谈。

预测窗口设为 H=16，即当前快照及后续 16 个快照；名义前视长度为 4 s，最大缓存区间为 4.25 s。它与候选路径的秒级参考发送时间相匹配。任务竞争可能使实际在途时间更长，默认仍允许覆盖不足的候选并记录其状态，mask 不构成服务保证。

### 2.6 接触场景先检查几何连通性

只把原 24 星、600 km 星座倾角提高到 80°，得到的全网连通时隙占比约为 66.5%；扩大为 66 星但保持 600 km 后约为 71.8%。这与稀疏几何、地球遮挡、纬度限制和选边规则共同有关，不能当作正常连通网络的主场景。

随后按几何条件检查 780 km / 6400 km 和 1200 km / 7800 km 两组高度与距离范围。两者在 5 s 间隔几何探查中均连通，选用较低的 780 km 案例后，再以完整 0.25 s 快照验证连通率为 100%。该筛选仅依据几何连通性，没有使用算法排名。全部中间审计保留于结果目录。

[Iridium 官方说明](https://www.iridium.com/network) 给出了约 780 km 的近极轨网络背景，但本场景采用 80° 倾角和 Walker-Delta 全圆 RAAN 分布，不是 Iridium 的实际轨道部署。链路变化也包含几何规则下的选边切换，不能全部称为地球遮挡造成的物理接触消失。

## 3. 已完成的预检查

采用与默认训练、验证、测试集合分离的诊断 seed=50、51、52。每个 seed 按任务序列等间隔抽取 256 个任务，检查全部保留的远程候选；检查使用空 CPU/在途队列和参考速率，不选择路由策略。

| 场景 | 全网期望 CPU 负载范围 | 最大逐星本地期望负载范围 | 连通时隙占比 | 候选发送区间检测到断链的比例 |
| --- | ---: | ---: | ---: | ---: |
| 匹配低负载对照 | 0.087–0.098 | 0.254–0.362 | 100% | 0.129%–0.268% |
| 通信—计算耦合 | 0.348–0.390 | 1.015–1.446 | 100% | 0.020%–0.101% |
| 计算密集 | 0.521–0.585 | 1.523–2.169 | 100% | 0.020%–0.101% |
| 链路受限 | 0.348–0.390 | 1.015–1.446 | 100% | 0.090%–0.322% |
| 高倾角接触 | 0.370–0.382 | 1.248–1.501 | 100% | 0.476%–0.744% |

以上 15 个场景/seed 组合的乐观期限不可行比例均为 0。抽样任务均有至少一个空系统下符合当前预测条件的候选；这不是实际共享服务下的成功保证。

候选比例的分母是全部抽样远程候选，不是任务数，也不是策略实际选择的路由。低负载对照和主场景抽到的任务不同，不能根据两者候选比例比较 future 收益。接触场景有可检测的风险，但比例仍较小；实际贡献必须由所选路径结果及独立训练消融检验。

原始文件位于 `results/scenario_preflight_20261005/`：`control`、`coupled`、`compute`、`link` 与最终的 `contact66_780`。`contact` 和 `contact66` 是连通性不足的中间配置记录，`geometry_probe.json` 保存几何探查。各目录的 `resolved_config.yaml` 与 SHA256 用于识别实际配置，不能把中间目录当作最终接触场景。

### 3.1 单 seed 启发式试跑：不作为论文结论

主场景采用诊断 seed=50、相同 9737 个任务与 CPU/拓扑轨迹，已完成以下检查：

| 方法 | 平均完成时延 / s | P95 完成时延 / s | 期限内成功率 |
| --- | ---: | ---: | ---: |
| local | 6.189 | 52.936 | 85.95% |
| computing_aware | 0.7774 | 1.6214 | 99.93% |
| computing_aware_future | 0.7776 | 1.6225 | 99.92% |

三种方法均无截尾和路由失败。future 在这次试跑中没有改善结果，计算感知方法的成功率仍接近饱和；因此不能声称新主场景已证明未来预测或 GAT 的优势。主场景用于检验通信与计算耦合，计算密集、链路受限和接触场景分别检验相应机制，均需正式训练和多 seed 对比。GAT/PPO 尚未在这些新场景训练。

结果保存在 `coupled_pilot50/summary.csv`，每个启发式 episode 的墙钟耗时约 106–133 s。这只是当前机器上的试跑耗时，不是 PPO 训练时间估计。66 星的额外策略试跑因耗时停止；`contact_pilot50` 仅保留输入与校准，`RUN_STATUS.md` 标记了中止状态，不得将其作为完整实验结果。完整的接触场景物理审计仍位于 `contact66_780`。

## 4. 复核命令

在工程目录的同一个 PowerShell 窗口执行：

```powershell
Set-Location E:\postgraduateLife\paper2\leo_compute_routing
$projectPython = (Resolve-Path .\.conda-env\python.exe).Path
$scenarioRun = "results/coupled_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')"

& $projectPython scripts/audit_route_exposure.py --config configs/experiments/walker_coupled.yaml --seeds 50 51 52 --max-tasks 256 --output "$scenarioRun/preflight_main"
& $projectPython scripts/audit_route_exposure.py --config configs/experiments/walker_contact_dynamic.yaml --seeds 50 51 52 --max-tasks 256 --output "$scenarioRun/preflight_contact"
```

该脚本新增于 `scripts/audit_route_exposure.py`。它不会训练、选择 proposed 策略或读取未来任务。

## 5. 主场景训练与测试

建议先各训练 10 updates，观察验证结果与执行耗时，再按照统一预算进行正式训练。

```powershell
& $projectPython scripts/train_ppo.py --config configs/experiments/walker_coupled.yaml --set rl.encoder=mlp --set rl.updates=10 --set rl.device=cuda --output "$scenarioRun/mlp_quick"
& $projectPython scripts/train_ppo.py --config configs/experiments/walker_coupled.yaml --set rl.updates=10 --set rl.device=cuda --output "$scenarioRun/gat_quick"
& $projectPython scripts/evaluate_ppo.py --checkpoints "$scenarioRun/mlp_quick/best.pt" "$scenarioRun/gat_quick/best.pt" --seeds 201 202 203 204 205 --output "$scenarioRun/quick_test"
& $projectPython scripts/plot_training.py "$scenarioRun/gat_quick"
```

600 s 到达阶段比原默认 60 s 长十倍。每个 episode 还可能包含排空阶段，新配置训练会明显更慢，不要把“10 updates”误认为很短的固定墙钟任务。现有低负载检查点直接测试只能作为迁移检查，正式主对比需在新场景重新训练。

正式训练、多初始化和模块消融仍使用 [完整实验命令](EXPERIMENTS.md) 的流程，将相应训练配置改为 `configs/experiments/walker_coupled.yaml`。例如：

```powershell
& $projectPython scripts/train_ppo.py --config configs/experiments/walker_coupled.yaml --set rl.device=cuda --output "$scenarioRun/gat_full"
& $projectPython scripts/train_ppo.py --config configs/experiments/walker_coupled.yaml --set rl.use_future=false --set rl.device=cuda --output "$scenarioRun/gat_no_future"
& $projectPython scripts/evaluate_ppo.py --checkpoints "$scenarioRun/gat_full/best.pt" "$scenarioRun/gat_no_future/best.pt" --seeds 201 202 203 204 205 --output "$scenarioRun/future_test"
```

## 6. 场景族与敏感性范围

对每个场景分别训练 MLP/GAT，并在相同测试 seed 比较。以下示例使用一次初始化，正式实验需按完整实验文档扩为多个初始化。

```powershell
foreach ($scenarioName in @('walker_low_load_control', 'walker_coupled', 'walker_compute_heavy', 'walker_link_heavy', 'walker_contact_dynamic')) {
    $scenarioConfig = "configs/experiments/$scenarioName.yaml"
    & $projectPython scripts/train_ppo.py --config $scenarioConfig --set rl.encoder=mlp --set rl.device=cuda --output "$scenarioRun/${scenarioName}_mlp"
    if ($LASTEXITCODE -ne 0) { throw "MLP training failed: $scenarioName" }
    & $projectPython scripts/train_ppo.py --config $scenarioConfig --set rl.device=cuda --output "$scenarioRun/${scenarioName}_gat"
    if ($LASTEXITCODE -ne 0) { throw "GAT training failed: $scenarioName" }
    & $projectPython scripts/evaluate_ppo.py --config $scenarioConfig --checkpoints "$scenarioRun/${scenarioName}_mlp/best.pt" "$scenarioRun/${scenarioName}_gat/best.pt" --seeds 201 202 203 204 205 --output "$scenarioRun/${scenarioName}_test"
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $scenarioName" }
}
```

预先定义的邻域范围可以采用：主场景到达率每时隙 2、4、6；任务业务容量 100、200、500 Mbit/s；期限区间 [1.5,6]、[2,8]、[3,10] s；名义预测长度 2、4、8 s。每个点保留结果并审计负载，按资源条件解释效果变化。期限更严格时重新报告最快 CPU 下界，不只报告方法的相对收益。

场景之间比较时延和成功率等实际指标，不能根据不同任务数或归一化量下的原始总 reward 排名。若未来模块在正式轨道实验中的贡献很小，应报告其作用范围；若 GAT 不优于 MLP，则需检查图表示、训练稳定性和非局部竞争信息是否提供了额外价值。
