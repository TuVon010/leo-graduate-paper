# LEO 卫星边缘计算研究

第二研究点采用分层协同方法：GAT-PPO 决定计算卫星，预测接触感知图路由决定到该节点的路径，KKT 分配链路和星上 CPU 资源。

- [实验工程与运行说明](leo_compute_routing/README.md)
- [中文系统模型](leo_compute_routing/docs/SYSTEM_MODEL_ZH.md)
- [英文系统模型](leo_compute_routing/docs/SYSTEM_MODEL.md)
- [服务器实验命令](leo_compute_routing/docs/SERVER_EXPERIMENTS.md)
- [研究方案](LEO_research_system_model_and_experiments.md)
- [工程结构](LEO_experiment_code_architecture.md)

旧联合节点/路径方法和配置已移除，历史实验数据保留。训练使用固定 seed，性能结论需以新版正式训练和一致对照为依据。
