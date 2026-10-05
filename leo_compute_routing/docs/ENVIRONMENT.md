# 环境与依赖配置说明

这份说明包含第一阶段工程所需依赖、Conda 创建命令和运行方法。项目源码要求 Python 3.9 及以上；建议新建 Python 3.11 环境。仿真只用 CPU，不需要 CUDA。

## 依赖

| 用途 | 包 | 版本约束 |
|---|---|---|
| 数组与数值运算 | NumPy | `>=1.20` |
| 图和路径搜索 | NetworkX | `>=2.6` |
| YAML 配置 | PyYAML | `>=6.0` |
| 测试 | pytest | `>=6.2` |
| 实验绘图 | matplotlib | `>=3.4` |

纯仿真不需要 PyTorch。新增 MLP/GAT-PPO 使用 PyTorch，依赖见 [requirements-rl.txt](../requirements-rl.txt)，本机选用的 CUDA 构建见 [requirements-rl-cu126.txt](../requirements-rl-cu126.txt)。不需要 PyTorch Geometric、Gymnasium、SGP4、MATLAB、STK 或 NS-3。核心依赖来自 [requirements.txt](../requirements.txt)，测试和绘图来自 [requirements-dev.txt](../requirements-dev.txt)。安装与训练说明见 [RL.md](RL.md)。

## 在本工程下创建独立环境

在 Anaconda Prompt 或已初始化 Conda 的 PowerShell 执行：

```powershell
Set-Location E:\postgraduateLife\paper2\leo_compute_routing
conda create --prefix .conda-env python=3.11 pip -y
conda run --prefix .conda-env python -m pip install -r requirements-dev.txt
```

`--prefix` 会将环境放在工程目录 `.conda-env`，不受这台电脑自定义的 Conda 环境目录影响。以下命令无需先激活环境：

```powershell
conda run --prefix .conda-env python --version
conda run --prefix .conda-env python -m pip list
conda run --prefix .conda-env python -m pytest -q
conda run --prefix .conda-env python scripts/run_baselines.py --config configs/smoke.yaml --output results/smoke
```

若 PowerShell 已完成 Conda 初始化，也可激活后直接运行：

```powershell
conda activate .\.conda-env
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/run_baselines.py --config configs/smoke.yaml --output results/smoke
```
### 方案 2：用【完整绝对路径】激活（如果你一定要进入环境）

复制这条命令直接运行：

```
conda activate E:\postgraduateLife\paper2\leo_compute_routing\.conda-env
```

✅ 成功后，提示符会变成 `(E:\postgraduateLife\paper2\leo_compute_routing\.conda-env)`，代表进入这个本地环境。

> 
> 原理：老版本 conda 激活外部环境，**必须写完整绝对路径，不能简写 .conda-env**。
Smoke 场景成功后，再运行 24 颗卫星的默认轨道近似场景：

```powershell
conda run --prefix .conda-env python scripts/run_baselines.py --config configs/base.yaml --output results/walker24
```

只需要运行仿真而暂时不用测试和绘图时，安装核心依赖即可：

```powershell
conda run --prefix .conda-env python -m pip install -r requirements.txt
```

## 保存依赖版本

确认环境可用并选定论文实验配置后，可以保存实际安装的版本：

```powershell
conda run --prefix .conda-env python -m pip freeze > requirements-lock.txt
conda env export --prefix .conda-env --no-builds > environment.yml
```

项目 `.gitignore` 已忽略 `.conda-env`，不要将环境目录提交到版本库。

## 资源需求

普通 CPU 即可。基础 24 星配置的拓扑缓存约十几 MB；48 和 72 星场景的拓扑数组随星数平方增长。任务量、负载、候选路径数量也影响耗时，建议先运行 `smoke.yaml`。

## 本机环境与创建状态

已有 Anaconda 的 Python 是 3.9.7；之前测试使用的 NumPy 1.20.3、NetworkX 2.6.3、PyYAML 6.0 和 pytest 6.2.4。之前生成的实验输出和 Python 缓存已清理，原有 `base` 和其他 Conda 环境保留。

此前自动创建尝试因访问包源失败而未完成。2026-10-04 本次项目复核已检测到工程下现有 `.conda-env`，Python 为 3.11.17；在该环境完成 55 项测试和五 seed 预实验，本次未安装或修改环境包。具体运行版本记录在实验 `manifest.json`，不需要为本次新增的审计、批次基线与统计安装额外依赖。

2026-10-05 用户授权在该项目环境安装 PyTorch。选择 RTX 4060 对应的官方 CUDA 12.6 构建；没有创建另一个 Conda 环境或修改其他已有环境。安装与当前阶段验证结果见 [RL.md](RL.md) 和 [VALIDATION.md](VALIDATION.md)。
