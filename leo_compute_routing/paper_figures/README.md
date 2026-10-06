# 系统模型图

本图按当前系统模型和代码绘制，将物理星座、接触窗口及决策流程放在同一张图中。图中的卫星位置、负载条和时间条为示意，未使用实验轨迹或测量值。

![中文系统模型图](system_model_zh.png)

## 文件

| 用途 | 中文版 | 英文版 |
| --- | --- | --- |
| 查看、汇报 | [PNG](system_model_zh.png) | [PNG](system_model_en.png) |
| 论文排版，矢量格式 | [PDF](system_model_zh.pdf) | [PDF](system_model_en.pdf) |
| 在矢量编辑器中修改 | [SVG](system_model_zh.svg) | [SVG](system_model_en.svg) |

PDF 已嵌入字体并完成渲染检查；SVG 保留可编辑文本，在其他机器上打开时需要对应字体。内容较多，论文双栏模板建议跨双栏排版。

## 怎么读这张图

1. **左上：任务和网络。** 输入已在蓝色源卫星，任务可以直接在源卫星计算，也可以沿蓝色路径逐跳传到绿色计算卫星。灰色线为其他当前 ISL。橙色卫星表示其他任务造成的计算竞争；工作量条的填充程度示意剩余工作量，不能读成 CPU 容量或实际利用率。图中 $D_u$ 为输入比特数，$C_u$ 为每比特计算周期数，$\tau_u$ 为期限；到达时刻为当前决策时隙，图中省略 $t_u$。
2. **左下：接触窗口。** 链路当前存在，不代表任务能够在断开前发送完。绿色条表示预测可用窗口，蓝色或红色条表示预计发送区间。红色候选超过窗口，因此被预测筛除。
3. **右侧：如何决策。** 利用允许读取的短期轨道快照建立接触计划，生成计算节点与无环路径候选；Shield 检查窗口、容量、期限和预测覆盖范围；任务感知 GAT-PPO 在候选之间选择；局部 KKT 规则分配共享链路和 CPU。紫色回路表示同批任务逐个选择后重新建立预测预约，后续任务会考虑前面任务的选择。
4. **底部：如何执行。** 任务整份输入按存储转发方式逐跳发送，经过传播后进入下一跳，全部输入到达目的卫星后进入共享 CPU。本地任务直接进入源卫星 CPU；未完成的传输和计算工作量跨时隙保留。

本系统的核心是：**联合选择“在哪里计算”和“走哪条路”，减少通信时间、计算竞争和接触中断带来的损失。**

## 与当前实现保持一致的范围

- 每个任务选择一个计算目的节点和一条固定路径，不拆分任务；本地执行是零跳候选。
- 逻辑控制器用于表达状态与决策信息流，并未假设额外 GEO/LUAV 控制节点。紫色、绿色虚线分别是状态和动作信息流，不是星间业务数据链路。
- 候选在当前图中搜索，预测用于评估未来服务；当前实现不等待断开的链路重新出现，也不进行途中重路由。
- 预约日历用于预测筛选和批次竞争估计；实际服务仍使用动态共享资源，Shield 不构成严格的期限成功保证。
- 策略采用变长候选评分，未绑定卫星 ID 输出维度；跨规模测试能力需要通过实验验证，图中不声称已取得泛化收益。
- 地面接入、结果回传和能耗不在当前代价模型中，图以源卫星为起点。

## 建议图注

**中文：** 接触窗口感知的 LEO 星上计算与路由系统。任务可在源卫星执行，或经多跳 ISL 送往选定计算卫星；逻辑控制器结合接触计划、预测可行性筛选、任务感知图策略和局部资源分配，联合决定计算目的节点与转发路径。图示拓扑、工作量和接触窗口仅用于说明机制。

**English:** Contact-aware LEO computing and routing system. Tasks are executed locally or forwarded over multiple ISLs to selected computing satellites. A logical controller combines contact planning, predictive feasibility screening, task-aware graph policy decisions, and local resource allocation to jointly select computing destinations and forwarding paths. Topology, workloads, and contact windows are illustrative.

## 参考画法与来源

参考 [Cao et al., *Computing-Aware Routing for LEO Satellite Networks: A Transmission and Computation Integration Approach*, Fig. 2](https://arxiv.org/abs/2211.08820) 中物理星座与网络抽象并列表达的组织方式；在本图中加入当前项目的计算工作量、接触窗口及方法流程。图形由本项目脚本原创绘制，未复制论文插图，亦未采用该论文的任务拆分或结果回传模型。

接触计划概念可进一步参阅 [NASA, Contact Graph Routing](https://ntrs.nasa.gov/citations/20120006508)。该引用解释接触窗口概念，并不表示当前实现已经包含完整 DTN 等待路由。

## 重绘

在工程根目录、已配置的 Python 环境中运行：

```bash
python scripts/draw_system_model.py
```

仅重绘英文：

```bash
python scripts/draw_system_model.py --language en
```

脚本：[draw_system_model.py](../scripts/draw_system_model.py)。依赖为 matplotlib 和 NumPy；中文图还需要中文字体，支持 Microsoft YaHei、Noto Sans CJK SC、SimHei 或 SimSun。图中的布局、标注与颜色均可在脚本中修改。
