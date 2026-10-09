# 系统建模图与方法流程图

系统模型使用纯物理与服务过程图；算法流程单独用于方法章节。新增图不读取实验数据，连接、时间、工作量和服务速率均为明确标注的示意。

## 系统模型用图

| 图 | 中文 PNG / SVG | 英文 PNG / SVG |
|---|---|---|
| 物理系统与任务服务 | [PNG](physical_system_zh.png) / [SVG](physical_system_zh.svg) | [PNG](physical_system_en.png) / [SVG](physical_system_en.svg) |
| E-1：动态连接与资源状态 | [PNG](dynamic_graph_zh.png) / [SVG](dynamic_graph_zh.svg) | [PNG](dynamic_graph_en.png) / [SVG](dynamic_graph_en.svg) |
| E-2：接触窗口与逐跳服务 | [PNG](contact_windows_zh.png) / [SVG](contact_windows_zh.svg) | [PNG](contact_windows_en.png) / [SVG](contact_windows_en.svg) |

E-1 中蓝色实线表示活动边，红色虚线表示用于对照的不可用边，后者不属于当前边集合。节点 Q 和 I 分别是驻留与在途 cycles；下方服务曲线是独立共享示例。

E-2 的时隙长度示例为 1 s，链路预算为 20 Mbit/s，10 Mbit 输入独占发送需要 0.5 s。每跳后传播 0.02 s；任务 u 在第二跳输入全部到达后进入 CPU。任务 v 在 t=3 s 接触结束时未完成发送，虚线框表示无法获得的服务。观察边界之外标记为未知，不当作实际断链时刻。示例参数不替代 compute24/contact66 的实验参数。

新图对应 [中文系统模型](../docs/SYSTEM_MODEL_ZH.md) 和 [英文系统模型](../docs/SYSTEM_MODEL.md)。重绘命令：

~~~bash
python scripts/draw_dynamic_network_model.py --language both
~~~

SVG 将文字转换为路径，便于没有同名字体的服务器或论文排版软件使用；可编辑重绘源代码在 scripts/draw_dynamic_network_model.py。

## 方法章节用图

以下原有文件包含 GAT-PPO、图路由与 KKT 流程，用于 [METHOD.md](../docs/METHOD.md)，不再放入系统模型正文。

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
