# 中英文系统模型图

图已与 feature schema=3 的分层实现一致：当前状态与有限接触计划 → GAT 图编码 → PPO 选择计算卫星 → 针对该节点的预测接触感知图路由 → KKT 链路/CPU 共享。

- [中文版 PNG](system_model_zh.png)、[SVG](system_model_zh.svg)、[PDF](system_model_zh.pdf)
- [英文版 PNG](system_model_en.png)、[SVG](system_model_en.svg)、[PDF](system_model_en.pdf)

论文图注可用：**Hierarchical LEO satellite edge computing architecture. Graph PPO selects the computing destination, predictive contact-aware routing determines a path to that destination, and KKT allocation shares communication and computation budgets.**

中文图注：**LEO 卫星边缘计算分层协同架构。图 PPO 选择计算目的节点，预测接触感知图路由决定到该节点的路径，KKT 分配通信与计算资源。**

左侧轨道、卫星和接触窗口均为原创建模示意，不是实际轨道快照或实验测量结果。蓝色路径表示计算节点选择后由图算法确定的整输入传输；源卫星也可本地计算。右侧批次回环表示预测预约，仅影响决策，不锁定真实资源。物理拓扑由预定连边规则生成，PPO 不选择 ISL 连接。

重绘：

```bash
python scripts/draw_system_model.py --language both
```

需 matplotlib 和中文字体；Windows 默认查找 Microsoft YaHei，Linux 可使用 Noto Sans CJK SC。使用 SVG/PDF 插图可保留矢量线条，PNG 用于预览。
