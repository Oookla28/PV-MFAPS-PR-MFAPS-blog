# 阅读路径

这份记录从植物病害检测开始，沿着三个问题推进。

第一，为什么固定 encoder 后可以读出类别，联合训练却收到不同的反馈。这里进入 SPD 协方差表示、谱函数和 Loewner 差商。

第二，为什么长尾训练中增加内循环步数并不总能提高尾部泛化。这里进入收缩率、稳定性风险包络、有限动作和离散 DP。

第三，复杂动作策略怎样在真实 GPU 上执行。这里进入 Fused graph、物理 forward 计数、CUDA event 和 measured cost。

正文已经把关键公式、插件矩阵、DRO case study、Fused 等价性和最终 HELD 结果放在同一条阅读路径上。深入章节只保留三类内容。

- [反馈几何与机制推导](01-origin-and-feedback-geometry.md)
- [PR-PV-MFAPS 数学推导](03-pr-pv-mfaps-derivation.md)
- [实验与消融记录](04-experiments-and-ablations.md)

## 图表索引

- `figures/bias-stagnation.svg` 展示 (ho\to1) 时偏差收益变平和稳定性经验代理增长。
- `figures/dro-action-flips.svg` 展示 Class 38 与 Class 68 的分区风险曲线。
- `figures/risk-curvature-heatmap.svg` 展示配对频率条件下的风险曲率。
- `figures/fused-co-design.svg` 展示 Legacy 与 Fused 的物理执行流。
- `figures/pareto-frontier.svg` 展示插件结果和最终 PR 结果的 measured Macro–Tail 位置。
