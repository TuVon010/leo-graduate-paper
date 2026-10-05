# 实验命令与论文完成度

**方法升级（2026-10-05）：** 接触窗口候选、预约日历 Shield 与跨规模冻结评估已经实现；最新方法命令和消融见 [CONTACT_METHOD.md](CONTACT_METHOD.md)。新推荐入口为 `configs/contact_ppo.yaml` 和 `configs/experiments/contact48_ppo.yaml`；旧 schema-1 检查点不能在当前网络加载，需要重新训练。下文通用实验流程仍有效，但低负载/KSP 配置与历史结果不能当作升级方法的正式数据。

当前代码可以开展正式实验，但已保留的 RL 数据主要是 2-update 合成场景功能检查，尚不足以支持“已收敛”“提出的方法优于基线”或“未来预测在轨道场景有效”的论文结论。可以开始撰写系统模型和方法，实验结论需要下面的训练、对比与消融结果。

**场景更新：** 已新增通信—计算耦合、计算密集、链路受限与高倾角接触配置，参数依据、预检查和对应训练命令见 [SCENARIOS.md](SCENARIOS.md)。下文的 `base.yaml` / `ppo.yaml` 命令仍对应原低负载场景；采用新主场景时，改用 `configs/experiments/walker_coupled.yaml`，并在该场景重新训练全部学习方法。

本文件列出当前实现可执行的实验命令。所有命令均在 **Windows PowerShell** 中运行，不需要重新安装 PyTorch。建议先执行第 0 节，然后执行第 2 节查看轨道场景的初步效果；正式实验再执行第 3–7 节。第 1、8、9 节用于功能检查或补充诊断。

## 0. 公共准备：在同一个 PowerShell 窗口执行

```powershell
Set-Location E:\postgraduateLife\paper2\leo_compute_routing
$projectPython = (Resolve-Path .\.conda-env\python.exe).Path
$experimentRoot = "results/experiments_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')"
New-Item -ItemType Directory -Path $experimentRoot | Out-Null

# 调用当前项目 Python；命令失败时停止当前命令块，避免继续使用缺失结果。
function Run-Python {
    & $projectPython @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python 命令失败，退出码：$LASTEXITCODE"
    }
}

$testSeeds = @(201, 202, 203, 204, 205, 206, 207, 208, 209, 210)
$initializationSeeds = @(2026, 2027, 2028)

Run-Python -c "import sys, torch; print(sys.executable); print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

后续命令使用 `Run-Python` 和上述变量，需要在同一窗口执行。该函数等价于调用项目环境的 `python.exe`，没有修改系统 Python。新开窗口时可以重新执行本节并使用新结果目录。

以下训练命令显式选择 `cuda`。当前机器已经验证 CUDA 可用；若换到无 GPU 机器，可将 `rl.device=cuda` 改为 `rl.device=cpu`。评估命令默认使用 CPU，以统一推理计时设备。当前 24 星网络较小、逐任务调用较多，GPU 不保证比 CPU 快。

不要删除已产生的结果。训练、RL 评估和多 seed 比较会拒绝非空输出目录；重跑时使用新目录。第 8、9 节的单次脚本没有同样完整的防覆盖检查，因此也应使用独立目录。

## 1. 功能检查：可选，不作为论文性能实验

### 1.1 仿真与 RL 测试

```powershell
Run-Python -m pip check
Run-Python -m pytest -q
```

### 1.2 六节点基线检查

```powershell
Run-Python scripts/run_baselines.py --config configs/smoke.yaml --output "$experimentRoot/baseline_smoke"
Run-Python scripts/plot_results.py "$experimentRoot/baseline_smoke/summary.csv"
```

### 1.3 两次更新的 MLP/GAT 训练及独立测试

```powershell
Run-Python scripts/train_ppo.py --config configs/rl_smoke.yaml --set rl.encoder=mlp --set rl.device=cpu --output "$experimentRoot/mlp_smoke"
Run-Python scripts/train_ppo.py --config configs/rl_smoke.yaml --set rl.device=cuda --output "$experimentRoot/gat_smoke"
Run-Python scripts/evaluate_ppo.py --checkpoints "$experimentRoot/mlp_smoke/best.pt" "$experimentRoot/gat_smoke/best.pt" --seeds 201 202 --output "$experimentRoot/smoke_test"
Run-Python scripts/plot_training.py "$experimentRoot/gat_smoke"
Run-Python scripts/plot_results.py "$experimentRoot/smoke_test/seed_201/summary.csv"
```

`smoke.yaml`、`rl_smoke.yaml` 与 `contact_stress.yaml` 都是合成拓扑配置。它们适合验证训练、执行和断链逻辑，不能充当物理轨道场景的性能证据。

## 2. 先看实际效果：24 星轨道预实验

### 2.1 参数审计

```powershell
Run-Python scripts/audit_scenario.py --config configs/base.yaml --output "$experimentRoot/base_audit.json"
Run-Python scripts/audit_scenario.py --config configs/base.yaml --set tasks.arrival_rate_per_slot=24.0 --output "$experimentRoot/load24_audit.json"
Run-Python scripts/audit_scenario.py --config configs/base.yaml --set tasks.arrival_rate_per_slot=48.0 --output "$experimentRoot/load48_audit.json"
Run-Python scripts/audit_scenario.py --config configs/base.yaml --set topology.link_capacity_bps=50000000.0 --output "$experimentRoot/link50m_audit.json"
```

审计会输出全网和逐星计算负载、独占最快 CPU 下仍无法满足期限的任务比例，以及预测窗口内当前链路消失的比例。窗口内存在链路变化，不代表所选任务实际遭遇断链；该审计也不会自动校准参数。

### 2.2 MLP 与 GAT 各训练 10 updates

```powershell
Run-Python scripts/train_ppo.py --config configs/mlp_ppo.yaml --set rl.updates=10 --set rl.device=cuda --output "$experimentRoot/mlp_quick"
Run-Python scripts/train_ppo.py --config configs/ppo.yaml --set rl.updates=10 --set rl.device=cuda --output "$experimentRoot/gat_quick"
```

每个 update 默认包含 2 个完整训练 episode，因此这里每种模型训练 20 个 episode。每个 episode 包含 240 个任务到达时隙，以及执行器实际需要的后续排空时隙。10 updates 用于看训练方向和排查问题，不说明已经收敛。

### 2.3 五个独立场景 seed，与现有五种基线一起比较

```powershell
Run-Python scripts/evaluate_ppo.py --checkpoints "$experimentRoot/mlp_quick/best.pt" "$experimentRoot/gat_quick/best.pt" --seeds 201 202 203 204 205 --output "$experimentRoot/quick_test"
Run-Python scripts/plot_training.py "$experimentRoot/mlp_quick"
Run-Python scripts/plot_training.py "$experimentRoot/gat_quick"
Run-Python scripts/plot_results.py "$experimentRoot/quick_test/seed_201/summary.csv"
```

评估默认包含 `local`、`shortest_offload`、`least_load`、`computing_aware` 和 `computing_aware_future`。同一评估 seed 下，各算法共享任务、CPU 和拓扑数据。

先查看 `quick_test/aggregate.csv` 的平均完成时延、P95、成功率、违约率、失败率和截尾率，再结合训练目录的 `validation.csv` 判断是否有稳定改善。`plot_results.py` 生成的是指定 seed 的探索图，不能代替多 seed 置信区间图。

## 3. 正式主对比：相同预算、多次训练初始化

以下为 **3 次初始化 × 2 种编码器 × 200 updates** 的第一轮正式实验计划。200 updates 是当前起始预算，不是收敛保证；训练曲线仍上升时，可按第 3.3 节在相同规则下增加预算。资源允许时，可在实验前将初始化 seed 扩至 5 个，例如 2026–2030。

### 3.1 训练

```powershell
foreach ($initializationSeed in $initializationSeeds) {
    Run-Python scripts/train_ppo.py --config configs/mlp_ppo.yaml --set "rl.seed=$initializationSeed" --set rl.device=cuda --output "$experimentRoot/mlp_$initializationSeed"
    Run-Python scripts/train_ppo.py --config configs/ppo.yaml --set "rl.seed=$initializationSeed" --set rl.device=cuda --output "$experimentRoot/gat_$initializationSeed"
    Run-Python scripts/plot_training.py "$experimentRoot/mlp_$initializationSeed"
    Run-Python scripts/plot_training.py "$experimentRoot/gat_$initializationSeed"
}
```

### 3.2 每次初始化分别在相同 10 个测试场景上评估

```powershell
foreach ($initializationSeed in $initializationSeeds) {
    Run-Python scripts/evaluate_ppo.py --checkpoints "$experimentRoot/mlp_$initializationSeed/best.pt" "$experimentRoot/gat_$initializationSeed/best.pt" --seeds $testSeeds --output "$experimentRoot/main_test_$initializationSeed"
}
```

三类 seed 含义不同：`rl.seed` 控制网络初始化和训练随机性；默认训练场景 seed 为 0–19；验证场景 seed 为 100–104；测试场景采用 201–210。一个场景 seed 同时改变任务到达和 CPU 抽样，因此当前结果并非“仅任务到达随机性”的统计。

`best.pt` 仅按验证集平均 reward 选择；不要依据测试集排名选择初始化或检查点。当前脚本输出每个初始化下的场景 seed 置信区间，尚没有自动汇总“训练初始化 × 测试场景”的分层统计。论文应保留全部初始化结果，再做跨初始化汇总，不能只选最好的一次。

若希望比较项目中已经存在的全部七种基线，可以将上面的评估命令改为：

```powershell
Run-Python scripts/evaluate_ppo.py --checkpoints "$experimentRoot/mlp_2026/best.pt" "$experimentRoot/gat_2026/best.pt" --seeds $testSeeds --algorithms batch_greedy_future batch_greedy local shortest_offload least_load computing_aware computing_aware_future --output "$experimentRoot/main_test_existing7_2026"
```

这只是启用已有策略，不新增或修改基线。`evaluate_ppo.py` 的配对差值以 `--algorithms` 中第一种基线为参照；默认参照是 `local`，上述可选命令的参照是 `batch_greedy_future`。应同时查看每种基线的结果，不能仅凭相对 local 的收益判断提出方法的价值。

### 3.3 断点恢复或延长训练

```powershell
Run-Python scripts/train_ppo.py --config "$experimentRoot/gat_2026/resolved_config.yaml" --resume "$experimentRoot/gat_2026/last.pt" --set rl.updates=400 --output "$experimentRoot/gat_2026_resume400"
```

这里的 400 是累计目标更新数，不是额外更新 400 次。恢复使用原目录的 `resolved_config.yaml`，避免遗忘原始 seed 或其他覆盖参数。若从 200 延长到 400，则继续执行 200 次更新。恢复目录保存续训日志，分析完整曲线时还需保留原训练目录。只允许改变目标 updates 和设备；改变任务负载、网络或模型参数应重新训练。

中断后仅能从已经保存的更新边界恢复。未完成的 episode 不包含在检查点中。

## 4. 方法消融：每个变体独立训练

主对比中的 MLP-PPO 已承担“替换 GAT 编码器”的消融。以下进一步分别关闭 future、mask 和同批预约。所有变体采用相同初始化 seed、训练/验证场景与 200-update 预算。

```powershell
$ablationVariants = @(
    @{Name = 'no_future'; Override = 'rl.use_future=false'},
    @{Name = 'no_mask'; Override = 'rl.use_mask=false'},
    @{Name = 'no_reservations'; Override = 'rl.use_reservations=false'}
)

foreach ($initializationSeed in $initializationSeeds) {
    foreach ($ablationVariant in $ablationVariants) {
        $ablationName = $ablationVariant.Name
        $ablationOverride = $ablationVariant.Override
        Run-Python scripts/train_ppo.py --config configs/ppo.yaml --set "rl.seed=$initializationSeed" --set rl.device=cuda --set $ablationOverride --output "$experimentRoot/${ablationName}_$initializationSeed"
        Run-Python scripts/plot_training.py "$experimentRoot/${ablationName}_$initializationSeed"
    }
    Run-Python scripts/evaluate_ppo.py --checkpoints "$experimentRoot/gat_$initializationSeed/best.pt" "$experimentRoot/mlp_$initializationSeed/best.pt" "$experimentRoot/no_future_$initializationSeed/best.pt" "$experimentRoot/no_mask_$initializationSeed/best.pt" "$experimentRoot/no_reservations_$initializationSeed/best.pt" --seeds $testSeeds --algorithms computing_aware_future --output "$experimentRoot/ablation_test_$initializationSeed"
}
```

多个 GAT 检查点的输出名称会自动追加后缀。本命令中的顺序为：完整 GAT、MLP、无 future、无 mask、无预约。使用每个 seed 的 `manifest.json` 中 `learned_policies` 的检查点路径识别变体，不要仅凭名称猜测。

`no_mask` 关闭候选掩码，同时关闭环境期限筛选；`no_future` 使预测窗口为零并取消未来拓扑特征。它们是当前实现所定义的完整模块消融，不是仅改变一行特征的实验。

## 5. 星座规模：零样本迁移与同规模训练

### 5.1 24 星训练模型直接迁移至 48/72 星

```powershell
foreach ($initializationSeed in $initializationSeeds) {
    Run-Python scripts/evaluate_ppo.py --config configs/medium.yaml --checkpoints "$experimentRoot/mlp_$initializationSeed/best.pt" "$experimentRoot/gat_$initializationSeed/best.pt" --seeds $testSeeds --output "$experimentRoot/transfer48_$initializationSeed"
    Run-Python scripts/evaluate_ppo.py --config configs/large.yaml --checkpoints "$experimentRoot/mlp_$initializationSeed/best.pt" "$experimentRoot/gat_$initializationSeed/best.pt" --seeds $testSeeds --output "$experimentRoot/transfer72_$initializationSeed"
}
```

这是冻结模型的跨规模迁移，不能称为“在 48/72 星场景中训练后的性能”。测试仍使用训练时固定的特征归一化尺度。

### 5.2 在 48/72 星场景分别重新训练

工程提供 `configs/experiments/walker48_ppo.yaml`、`walker72_ppo.yaml`，继承 PPO 参数，并按现有 medium/large 配置同步扩大任务到达率与热点集合。

```powershell
foreach ($satelliteCount in @(48, 72)) {
    $scaleConfig = "configs/experiments/walker${satelliteCount}_ppo.yaml"
    Run-Python scripts/audit_scenario.py --config $scaleConfig --output "$experimentRoot/scale${satelliteCount}_audit.json"
    foreach ($initializationSeed in $initializationSeeds) {
        foreach ($encoderName in @('mlp', 'gat')) {
            Run-Python scripts/train_ppo.py --config $scaleConfig --set "rl.encoder=$encoderName" --set "rl.seed=$initializationSeed" --set rl.device=cuda --output "$experimentRoot/${encoderName}_${satelliteCount}_$initializationSeed"
        }
        Run-Python scripts/evaluate_ppo.py --config $scaleConfig --checkpoints "$experimentRoot/mlp_${satelliteCount}_$initializationSeed/best.pt" "$experimentRoot/gat_${satelliteCount}_$initializationSeed/best.pt" --seeds $testSeeds --output "$experimentRoot/scale${satelliteCount}_test_$initializationSeed"
    }
}
```

规模实验保持每颗卫星的期望到达任务数和热点占比一致，拓扑与 CPU 抽样仍会变化。当前 GAT 使用稠密注意力，规模增加时还应报告完整决策开销。24 星与 48/72 星的 reward 归一化常数不同，不能将不同规模的原始 reward 直接作为性能排序。

## 6. 参数敏感性：冻结 RL 模型与基线共同测试

当前 `run_sensitivity.py` 只接受基线名称。评估 RL 的参数变化应生成测试配置，再调用 `evaluate_ppo.py`。下面命令扫描到达率、链路容量、CPU 容量、热点比例、期限、预测窗口和 K 条候选路径。

需要先完成第 3 节的训练。以下先使用初始化 2026 的冻结模型，完成初步扫描后，再对其余初始化重复。

```powershell
$sweepSpecifications = @(
    @{Name = 'arrival'; Section = 'tasks'; Field = 'arrival_rate_per_slot'; Values = @('4.0', '12.0', '24.0', '48.0')},
    @{Name = 'link'; Section = 'topology'; Field = 'link_capacity_bps'; Values = @('50000000.0', '100000000.0', '500000000.0', '1000000000.0')},
    @{Name = 'cpu'; Section = 'compute'; Field = 'cpu_cycles_per_second'; Values = @('[10000000000.0, 50000000000.0]', '[20000000000.0, 100000000000.0]', '[40000000000.0, 200000000000.0]')},
    @{Name = 'hotspot'; Section = 'tasks'; Field = 'hotspot_probability'; Values = @('0.0', '0.3', '0.5', '0.7')},
    @{Name = 'deadline'; Section = 'tasks'; Field = 'deadline_seconds'; Values = @('[0.25, 0.5]', '[0.5, 2.0]', '[1.0, 4.0]')},
    @{Name = 'lookahead'; Section = 'routing'; Field = 'lookahead_slots'; Values = @('0', '2', '8', '16')},
    @{Name = 'k_paths'; Section = 'routing'; Field = 'k_paths'; Values = @('1', '3', '5')}
)

foreach ($sweepSpecification in $sweepSpecifications) {
    $sweepName = $sweepSpecification.Name
    $sectionName = $sweepSpecification.Section
    $fieldName = $sweepSpecification.Field
    for ($valueIndex = 0; $valueIndex -lt $sweepSpecification.Values.Count; $valueIndex++) {
        $parameterValue = $sweepSpecification.Values[$valueIndex]
        $variantConfig = "$experimentRoot/sweep_${sweepName}_${valueIndex}.yaml"
        @"
extends: ../../configs/base.yaml
${sectionName}:
  ${fieldName}: $parameterValue
"@ | Set-Content -LiteralPath $variantConfig -Encoding utf8
        Run-Python scripts/audit_scenario.py --config $variantConfig --output "$experimentRoot/sweep_${sweepName}_${valueIndex}_audit.json"
        Run-Python scripts/evaluate_ppo.py --config $variantConfig --checkpoints "$experimentRoot/mlp_2026/best.pt" "$experimentRoot/gat_2026/best.pt" --seeds $testSeeds --output "$experimentRoot/sweep_${sweepName}_${valueIndex}"
    }
}
```

配置文件直接保存在 `$experimentRoot` 下，继承路径 `../../configs/base.yaml` 按这一目录层级解析。将文件移至其他目录时需要调整继承路径。

这些点是用于探索的工程参数，不是已校准的真实业务配置。高到达率或高热点比例可能产生过载，应保留并明确标注，结合 `calibration.json` 分析。不要按提出方法的最终排名筛选主场景。

上述实验测量的是冻结策略对参数变化的适应性。若要比较各参数下重新训练后的性能，应针对各点另建训练配置，并对全部学习方法采用一致预算重新训练。尤其是 H/K 的冻结模型扫描，不能替代第 4 节的独立训练消融。不同配置的任务 trace 可能不同，共同回放保证适用于同一参数点内各算法。

### 6.1 时隙粒度：保持物理到达率、时域与名义预测长度

以下比较 0.125 s 与 0.5 s 的快照粒度。每秒期望到达率保持 48 tasks/s，任务到达阶段保持 60 s，最大排空阶段保持 300 s，名义预测长度保持 2 s。

```powershell
$slotSpecifications = @(
    @{Name = 'dt0125'; Dt = '0.125'; Slots = '480'; Drain = '2400'; Arrivals = '6.0'; H = '16'},
    @{Name = 'dt0500'; Dt = '0.5'; Slots = '120'; Drain = '600'; Arrivals = '24.0'; H = '4'}
)
foreach ($slotSpecification in $slotSpecifications) {
    $slotName = $slotSpecification.Name
    $slotConfig = "$experimentRoot/$slotName.yaml"
    @"
extends: ../../configs/base.yaml
simulation:
  slot_seconds: $($slotSpecification.Dt)
  slots: $($slotSpecification.Slots)
  drain_slots: $($slotSpecification.Drain)
tasks:
  arrival_rate_per_slot: $($slotSpecification.Arrivals)
routing:
  lookahead_slots: $($slotSpecification.H)
"@ | Set-Content -LiteralPath $slotConfig -Encoding utf8
    Run-Python scripts/evaluate_ppo.py --config $slotConfig --checkpoints "$experimentRoot/mlp_2026/best.pt" "$experimentRoot/gat_2026/best.pt" --seeds $testSeeds --output "$experimentRoot/$slotName"
}
```

当前缓存覆盖当前快照及 H 个未来快照，因此完整覆盖长度为 `(H+1) × dt`，粒度变化时仍有一个时隙的覆盖差异。任务按边界批量到达，改变 dt 也会改变批次粒度，结果不能仅归因于轨道位置精度。

## 7. 更长轨道窗口与推理开销

### 7.1 延长轨道观测，检查预测是否有作用

```powershell
$longOrbitConfig = "$experimentRoot/long_orbit.yaml"
@"
extends: ../../configs/base.yaml
simulation:
  slots: 2400
"@ | Set-Content -LiteralPath $longOrbitConfig -Encoding utf8

Run-Python scripts/audit_scenario.py --config $longOrbitConfig --output "$experimentRoot/long_orbit_audit.json"
Run-Python scripts/evaluate_ppo.py --config $longOrbitConfig --checkpoints "$experimentRoot/gat_2026/best.pt" "$experimentRoot/no_future_2026/best.pt" --seeds 201 202 203 204 205 --algorithms computing_aware computing_aware_future --output "$experimentRoot/long_orbit_test"
```

此处将任务到达时域延长至 600 s，评估耗时和原始数据量也会增加。这是短时域训练模型的长时域迁移检查；正式主实验若采用该时域，需要相应独立训练，并对齐所有方法的预算与观测设置。出现更多拓扑变化本身不保证 future 有收益。

### 7.2 CPU/CUDA 推理对比

```powershell
Run-Python scripts/evaluate_ppo.py --checkpoints "$experimentRoot/mlp_2026/best.pt" "$experimentRoot/gat_2026/best.pt" --seeds 201 202 --device cpu --output "$experimentRoot/inference_cpu"
Run-Python scripts/evaluate_ppo.py --checkpoints "$experimentRoot/mlp_2026/best.pt" "$experimentRoot/gat_2026/best.pt" --seeds 201 202 --device cuda --output "$experimentRoot/inference_cuda"
```

当前每个 seed 的 `summary.csv` 包含 `policy_select_ms_per_task`、`observation_build_seconds` 和 `episode_wall_seconds`。前者仅统计策略选择，观测构建耗时另行记录，不能把它当完整决策耗时。GPU 浮点结果可能略有变化。这组命令是计时诊断，不提供包含预热、重复测量与 GPU 同步控制的正式性能基准；如要发表严格的实时性结论，还需专用计时实验。

## 8. 已有基线消融与敏感性脚本

### 8.1 七种基线，多 seed 与配对统计

```powershell
Run-Python scripts/run_multiseed.py --config configs/base.yaml --seeds $testSeeds --output "$experimentRoot/baselines_multiseed"
```

默认配对参照为 `batch_greedy`，与 RL 评估默认参照不同，应按输出文件的 reference 字段解释。

### 8.2 启发式模块消融

```powershell
Run-Python scripts/run_ablation.py --config configs/base.yaml --output "$experimentRoot/heuristic_ablation_walker"
Run-Python scripts/run_ablation.py --config configs/contact_stress.yaml --output "$experimentRoot/heuristic_ablation_synthetic"
```

包含完整启发式、无 future、CPU 与链路同时等分、无期限筛选。该脚本不是 GAT/PPO 消融，且同时改变 CPU 和链路分配不能单独归因某一类资源。

### 8.3 基线参数扫描

```powershell
Run-Python scripts/run_sensitivity.py --config configs/base.yaml --parameter tasks.arrival_rate_per_slot --values 4 12 24 48 --seeds 201 202 203 204 205 --output "$experimentRoot/baseline_arrival_sweep"
Run-Python scripts/run_sensitivity.py --config configs/base.yaml --parameter topology.link_capacity_bps --values 50000000 100000000 500000000 1000000000 --seeds 201 202 203 204 205 --output "$experimentRoot/baseline_link_sweep"
Run-Python scripts/run_sensitivity.py --config configs/base.yaml --parameter tasks.hotspot_probability --values 0.0 0.3 0.5 0.7 --seeds 201 202 203 204 205 --output "$experimentRoot/baseline_hotspot_sweep"
Run-Python scripts/run_sensitivity.py --config configs/base.yaml --parameter tasks.deadline_seconds --values '[0.25, 0.5]' '[0.5, 2.0]' '[1.0, 4.0]' --seeds 201 202 203 204 205 --output "$experimentRoot/baseline_deadline_sweep"
```

默认仅比较 local、computing_aware、computing_aware_future，可通过 `--algorithms` 指定其他已有基线。输出 `sensitivity.csv` 为逐参数、逐 seed 的原始汇总，当前此脚本不自动生成参数点置信区间。

## 9. 拓扑缓存、回放与单 seed 绘图

### 9.1 生成拓扑缓存并运行

```powershell
Run-Python scripts/generate_topology.py --config configs/base.yaml --output "$experimentRoot/topology_base.npz"
Run-Python scripts/run_baselines.py --config configs/base.yaml --topology-cache "$experimentRoot/topology_base.npz" --output "$experimentRoot/baseline_cached"
```

### 9.2 用保存的任务、拓扑与配置回放

```powershell
Run-Python scripts/run_baselines.py --config "$experimentRoot/baseline_cached/resolved_config.yaml" --topology-cache "$experimentRoot/baseline_cached/topology.npz" --task-trace "$experimentRoot/baseline_cached/task_trace.json" --output "$experimentRoot/baseline_replay"
Run-Python scripts/plot_results.py "$experimentRoot/baseline_cached/summary.csv"
```

CPU 由保存配置中的 seed 确定性重建。耗时指标可能不同，不能要求墙钟时间相同。`plot_results.py` 接受含每种算法一行的 `summary.csv`；**不要直接传入长表格式的 `aggregate.csv`**。

## 10. 看哪些文件，以及论文还缺什么

| 文件 | 作用 |
| --- | --- |
| `episodes.csv`、`updates.csv`、`validation.csv` | 原始训练、优化与验证记录；不同训练 episode 的场景可能不同 |
| `best.pt`、`last.pt`、`resolved_config.yaml`、`training_manifest.json` | 验证选模、恢复训练与复现 |
| `seed_metrics.csv` | 所有算法、所有测试场景的指标 |
| `aggregate.csv`、`aggregate.json` | 等权场景 seed 均值与 95% bootstrap CI |
| `paired_differences.csv` | algorithm 减 reference 的配对差；时延负值通常为改善，成功率正值为改善 |
| `seed_*/manifest.json` | task/CPU/topology 指纹及检查点 SHA256，识别消融变体 |
| `seed_*/calibration.json` | 当前参数点的负载与期限下界审计 |
| `seed_*/算法名/tasks.csv`、`slots.csv` | 逐任务与逐时隙记录，用于进一步统计与绘图 |

主指标需在正式测试前确定。建议同时报告平均完成时延、P95、按时成功率、违约率、路由失败率、截尾率及资源利用率。平均完成时延只对完成任务统计，不能单独代表方法好坏。测试集用于最终评估，不用于决定超参数、预算或选模；调参应依据验证集。

| 论文内容 | 当前状态 | 对论文的影响 |
| --- | --- | --- |
| 动态轨道、整任务卸载、存储转发、跨时隙竞争 | 已实现，已有测试 | 系统模型可以开始写，但需明确近似假设 |
| 候选 MLP/GAT-PPO、mask、预约、训练/恢复 | 已实现，短训练验证通过 | 方法章节可写，尚无收敛与优越性证据 |
| 正式轨道主对比、多个训练初始化 | 尚未完成 | 当前数据不足以支撑核心性能结论 |
| GAT/future/mask/预约独立贡献 | 可按命令开展，尚无正式结果 | 需证明模块贡献，不能仅报告完整方法排名 |
| 未来拓扑在物理轨道中的实际收益 | 尚未建立 | 默认场景短、失败少，预测卖点可能无法验证 |
| 参数来源和校准、所选路径接触风险与预测误差 | 只有部分审计 | 仍需参数依据及更细诊断 |
| 多初始化汇总、CI 论文图、逐资源负载及完整计时 | 部分数据已有，分析和绘图不完整 | 尚需整理可发表的结果与统计 |
| CPU/链路独立分配消融 | 当前只有统一 `resource.allocation` 开关 | 若要分别声称两类分配贡献，需要新增独立开关与实验 |
| 小规模 Oracle/最优参照 | 尚未实现 | 不是所有论文的必选项，但有助于量化方法提升空间 |

总体判断：**工程框架足以启动论文实验，现存实验结果尚不足以支撑完整论文的核心结论。** 最重要的是先验证完整 GAT-PPO 能否在多种合理负载下稳定改善，并检查各模块的独立贡献。若 MLP 或现有启发式更好，或者 future 无可测收益，应据此改进方法或收缩主张。GAT、PPO 和掩码的组合本身不能自动建立创新性，需要结合相关工作与实验说明本项目解决的具体困难。

本文件提供当前已经支持的实验命令；小规模 Oracle、独立 CPU/链路消融和未实现的论文图没有可用命令。正式实验耗时应先由第 2 节的本机运行记录估计，再确定预算。
