# leo-graduate-paper

硕士毕业论文「面向非地面网络边缘计算的通信计算协同优化方法研究」第二研究点：

**面向动态多 LEO 卫星边缘计算的计算感知路由与资源协同优化方法**

## 仓库内容

| 路径 | 说明 |
|---|---|
| `LEO_research_system_model_and_experiments.md` | 研究方案：系统模型、算法设计、实验规划 |
| `LEO_experiment_code_architecture.md` | 实验工程代码结构设计说明 |
| `leo_compute_routing/` | 实验仿真工程（动态拓扑 / 任务 / 路由 / 资源分配 / 基线 / 评估），详见其 `README.md` |

## 当前状态

- 已完成：事件驱动 LEO 边缘计算仿真环境、5 种基线策略、消融与敏感性扫描框架、44 项单元测试
- 进行中：环境参数校准、多 seed 论文级基线实验
- 规划中：MLP-PPO → GAT-PPO → Feasibility Mask 强化学习实现与实验
