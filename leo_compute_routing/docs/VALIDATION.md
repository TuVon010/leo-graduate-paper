# 新版实现验证记录

日期：2026-10-07。当前架构为计算卫星 GAT-PPO → 后置预测接触图路由 → KKT 资源共享，feature schema=3。旧联合动作、KSP、完成时间 logit 先验及旧配置已移除；既有 `results/` 数据和日志保留，历史说明见 [HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md)。

## 回归与接口

- `python -m pytest -q`：**114 passed**。覆盖原物理守恒/事件边界、双向共享、接触半开窗口、预测覆盖隔离、动态 CPU 服务、GAE 与概率快照、实际梯度、恢复与设备迁移。
- 新版测试验证节点动作数量等于卫星数、节点重编号等变、只保存请求节点概率、后置路由与回退记录、预约不改变执行器、拒绝惩罚与实际断链分开统计。
- 验证相同固定权重与特征尺度支持 24/48/72/96 星；旧 schema 权重和旧路由配置参数明确拒绝加载。
- 路由小图与穷举路径代价一致；扩展预算截断明确记录。加入到目标的最短跳数下界剪枝和带预约的路径前缀积分复用，验证与完整积分一致。

## 实际运行

| 检查 | 保存位置 | 说明 |
|---|---|---|
| GAT CPU 2 updates | `results/hierarchical_smoke_20261007/` | 六节点功能检查，保存训练/验证及 best/last |
| 服务器 main 全流程 | `results/hierarchical_runner_check_20261007/` | GAT、MLP 各 2 updates，单 seed 100 开发评估与基线，绘图、日志、角色映射和完成标记 |
| 最终评估配置 | `results/hierarchical_evaluation_finalcheck_20261007/` | 单 seed 100、GAT/MLP 与 7 基线，移除重复规则命名 |
| compute24 CUDA | `results/hierarchical_compute24_check_20261007/` | 16 到达时隙、1 update，RTX 4060 实际 GPU 前向/反向与保存 |
| contact66 CPU | `results/hierarchical_contact66_finalcheck_20261007/` | 16 到达时隙、1 update；验证汇总正确打印路由拒绝率 |
| 接触路由审计 | `results/hierarchical_audit_check_20261007/` | 16 时隙、4 个均匀抽样任务、seed 100；新逐目标审计 CSV 与校准 |

上述轨道短检查保持对应物理场景，只缩短时域和训练量，不作为正式性能实验。两个轨道检查均完成全部任务、无截尾和数值异常。短训练尚未收敛，尤其初始策略存在较多路由拒绝；不能据此认定 PPO 已优于贪心。

服务器 main/routing/modules/resource/scale/all/audit/sensitivity 共 8 组命令计划完成 dry-run 校验；剩余 YAML 配置及继承通过校验。单 seed 开发比较不生成置信区间，复用验证的角色在 study manifest 和控制台中明确标注。

## 图与文档

中英文系统模型、方法、工程结构、物理参数和服务器命令均同步至三层流程。PNG/SVG/PDF 系统图已重新生成，并将两份 PDF 用 Poppler 渲染后逐一检查，未发现裁切、重叠或乱码。数学式中的排队、在途与预测预约均对应实现，未将 Q/F 叠加到实际时延，也未声称参考预测保证成功。

现有 Python 3.11 / PyTorch 2.7.1+cu126 环境继续使用，此次未安装新包。控制台原文在相应 `console_logs/`，批量子日志在输出根目录 `logs/`。正式服务器训练应使用全新目录重新开始，不能恢复旧 20 updates 权重。
