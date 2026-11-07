# PV-MFAPS / PR-PV-MFAPS 研究叙事与博客结构

## 页面定位

这不是实验报告，也不把过程写成方法说明书。叙事从一个具体的长尾训练矛盾开始。读者先看到模型为什么“读得出来却学不对”，再看到几轮方法如何因为目标、预算和实现仍未对齐而不断收紧，最后进入 PV-MFAPS 的风险包络、离散分配和 PR-PV-MFAPS 的分区鲁棒版本。

全页使用宋体系字体。主页面适合电脑端纵向阅读，侧栏在手机端变成横向可滚动导航。数学使用 MathJax，长公式放在可横向滚动的公式卡片中。每个深层主题可以从主页面跳转到一个独立 Markdown 阅读页，阅读页用同一套 Markdown renderer 解析标题、表格、代码、blockquote、LaTeX 和锚点，返回按钮回到主页面原位置。

## 页面标题与开场

建议标题：`从反馈失配到分区鲁棒分配：PV-MFAPS 的一次长尾研究复盘`

开场句要有剧情感：

> 一开始，我们只是想让尾部类别多拿几步内循环更新。后来才发现，真正决定训练轨迹的，不是“谁更稀有”，而是一个读出误差经过几何映射之后，会把什么样的信号送回编码器。

首屏指标卡放四个数字。

- PlantSPD 69 类，FIT 4,184，HELD 1,046
- FIT 计数范围 5–344，头尾比 68.8
- PR-PV-MFAPS 24 个最终 matched artifacts
- PR 相对 fixed-k 的 HELD Macro-F1 +0.0288，Tail-10% +0.0317

同时放一条边界说明。`这组结果是 PlantSPD matched 结果，不是社区 SOTA 或跨任务泛化结论。`

## 导航章节

1. 起点：为什么一个分类头会改变研究结论
2. 数据：长尾和配对分区
3. 机制：前向可读出不等于反馈可用
4. 第一轮转折：mean、CVaR 和无约束极小极大
5. PV-MFAPS：风险包络与有限动作 DP
6. 系统优化：为什么 fused 路径必须单独证明
7. A100：动作空间、预算和 epsilon sweep
8. 分区异质性：从 gate 到 PR-PV-MFAPS
9. 最终结果：哪些门通过，哪些门仍然关闭
10. 代码、工件和可重复运行

## 1. 起点

故事从原始 Plant SPD/叶境智鉴工程进入，而不是直接从算法名开始。

重点写三层转向。

- 工程层先要求真实图像、真实标签、可部署推理和长尾数据。
- SPD 层把 token/patch 特征变成协方差矩阵，并比较 Raw、Log、Sqrt。
- 训练层发现分数差异可能来自表示、读出或优化轨迹。

核心引用框：

> 固定编码器后可以达到高可读性，联合训练却仍然可能收不到同一条反馈。这说明“信息存在”和“训练能利用它”是两个问题。

图表：`story-timeline.svg`，标记 PlantSPD → feedback mismatch → R-FAPS/TF-GPS → PV-MFAPS → fused → A100 → PR。

## 2. 数据与长尾

展示 FIT/HELD 协议、p0/p1、类别频率和配对关系。

必须说清楚：p0 与 p1 的 69 类 FIT counts 逐类完全相同。分区差异来自风险几何，不来自频率重新分配。

图表：

- 对数坐标的 69 类计数条形图
- p0/p1 counts 差异热图，显示全为 0
- FIT/HELD 单次读取时间线

Blockquote：

> 当两个分区拥有完全相同的类别频率时，分区鲁棒策略发生的动作变化不能用重采样解释。

## 3. 机制线

先写单头目标。

\[
L(\psi,W)=\frac1n\sum_i\ell(Wz_\psi(x_i),y_i)+\frac{\lambda}{2}\|W\|_F^2
\]

定义最优读出与实际读出。

\[
W^*(\psi)=\arg\min_W L(\psi,W),\qquad
 e_\psi=\nabla_\psi L(\psi,W)-\nabla_\psi L(\psi,W^*(\psi))
\]

解释为什么 `loss gap` 小不自动意味着反馈误差小。用 `Z U=0` 的前向等价而反向不等价例子。

再进入 SPD 映射。给出谱函数的 Loewner 差商。

\[
D\phi_C[H]=U\bigl(G_\phi(\Lambda)\odot(U^THU)\bigr)U^T
\]

强调 \(\phi(\lambda)\) 控制前向表示，\(G_{ij}\) 控制反馈方向，二者不相同。

图表：

- 前向 logits 相同、encoder gradient 不同的二维示意图
- Raw/Log 的 Loewner 差商热图
- `kappa` 与频率、loss、gradient norm 的偏相关散点图

## 4. 第一轮转折

把这一轮写成剧情转折，而不是结果表堆积。

建议并列四张卡。

- `mean TF-GPS`：均值调度使梯度方差爆炸，FIT/HELD 同时下降。
- `CVaR-TF-GPS`：尾部略有改善，但 Macro-F1 仍低于 effective-number。
- `old R-FAPS`：Tail-10% 较高，整体 Macro-F1 下降。
- `corrected K-MFAPS`：Tail-10% 达到 0.2778，但成本约 2.154e12 FLOPs，不满足 matched wall budget。

图表：Macro-F1–Tail-10% Pareto 散点图，点大小代表 FLOPs。把 native、effective-number、fixed-k、old RFAPS、CVaR、K-MFAPS、PC-FA 画在同一坐标。

Blockquote：

> 尾部优先、均值优先和均值调度，分别把三角形的三个角暴露出来。问题不是“再选一种权重”，而是如何在风险和预算之间分配内循环精度。

## 5. PV-MFAPS 数学

先给风险包络。

\[
R_c(k)\le b_c\rho_c^{2k}+s_c(k)+\beta_c(k)+e_c(\delta)
\]

解释四项来源。

- 偏差项：内层读出未收敛
- 方差项：有限批次和类别支持
- 稳定性项：强凸、光滑和 GD 收缩下的替换样本界
- 置信项：有限动作集合上的 simultaneous bound

写稳定性推导。

\[
\|W_k(S)-W_k(S')\|
\le \frac{2\eta G}{n(1-\rho)}(1-\rho^k)
\]

说明这是一条条件性稳定性界，不把经验中的 \(\gamma^k\) 方差爆炸写成无条件定理。

再写离散动作问题。

\[
\min_{k_c\in\{0,1,2,4\}}
 \sum_c w_cR_c(k_c)
\quad\text{s.t.}\quad
\sum_c q_ck_c\le B
\]

解释 DP 状态、量化成本、Pareto frontier 和残差界。

图表：

- 一个类的 (R_c(k),\beta_c(k),e_c(k)) 风险包络图
- DP frontier size 随 class prefix 增长图
- k 动作直方图和预算占用图

## 6. fused 系统线

把实现优化单独写成一条系统剧情。

展示 legacy 和 fused 的执行图。

- legacy 重复 encoder forward
- fused 单 encoder graph
- action sequence 和 logical counters 保持一致
- real-checkpoint state max abs delta 低于 1e-5
- physical encoder forwards 3→1

必须在页面上明确系统速度证据和算法收益证据分开。

图表：

- legacy/fused 流程图
- physical forwards 计数柱状图
- FLOPs/wall ratio 成本散点图

## 7. A100 与 epsilon sweep

描述从 3090 到 A100 的动作空间扩展。列出 Kmax、budget、gamma、batch 的 pilot grid，说明 one failed grid 进入 quarantine。

epsilon sweep 图表至少包含两条线。

- \(\Phi_\epsilon(R_0,R_1)\) 随 epsilon 的变化
- p0/p1 risk 随 epsilon 的变化

另加动作变化表。

- epsilon 0.1 → 0.01 只改变 2/69 类
- class 38: k=2→0
- class 68: k=0→1
- p0/p1 counts 完全相同

Blockquote：

> epsilon 是活的，但在当前动作集合里只改变了两个类。鲁棒目标有作用，然而 p1 的结构性风险不是单靠重新分配内循环步数就能消掉的。

## 8. 分区 gate 与 PR-PV-MFAPS

先说明 gate 只看 FIT。

- macro gap 0.01392，超过 0.01 阈值
- tail gap 0.00291，没有达到 0.02 阈值
- action TV 0.02899，没有达到 0.10 阈值
- macro gap 触发 PR registration

给出 KL ambiguity set。

\[
\mathcal U_\epsilon=\{u\in\Delta^2:D_{KL}(u\|\bar u)\le\epsilon\}
\]

\[
\Phi_\epsilon(v)=\sup_{u\in\mathcal U_\epsilon}u^Tv
=\inf_{\tau>0}\left[\tau\epsilon+	au\log\sum_p\bar u_p e^{v_p/\tau}\right]
\]

说明共享策略的重要性。p0/p1 不能分别求解，否则只是两套 partition-specific policy。

图表：

- nominal policy 与 robust policy 的动作差异图
- p0/p1 risk vector 与 KL worst-case mixture
- PR action allocation 的 class heatmap

## 9. 最终结果

用一个读者能一眼看懂的表。

| arm | FIT Macro | FIT Tail | HELD Macro | HELD Tail | wall | FLOPs |
|---|---:|---:|---:|---:|---:|---:|
| PR-PV-MFAPS | 0.9703 | 0.9655 | 0.4841 | 0.2976 | 0.606 | 1.077e12 |
| fixed-k | 0.9459 | 0.9429 | 0.4553 | 0.2659 | 0.663 | 1.257e12 |
| effective-number | 0.9668 | 0.9665 | 0.4760 | 0.2778 | 0.563 | 1.077e12 |

主结果只支持 PR vs fixed-k 的 matched gate。

再放一张“哪些结果不能直接升级”的卡片。

- relative effective-number wall ratio 1.0768
- no CIFAR-LT/ImageNet-LT execution
- no second SPD project
- no independent semantic review
- no community SOTA statement

## 10. 转折、修正与可复现

把每个边界写成研究过程中的“悬念”，但不要让证据边界压过主线。

可展开 details。

- eps=0.1 smoke 被 eps=0.01 policy selection 替换
- PR/BS internal arm state collision，18 artifacts 版本隔离
- first HELD path boundary，held_reads remained 0
- final clean PR/control split rerun

每个转折卡片要显示 `INVALID_CONFIG`、`REGISTERED_NOT_EXECUTED` 或 `UNVERIFIED` 标签，并提供原始 manifest 的下载链接。

## 11. 页面技术实现

建议仓库结构。

```
index.html
assets/
  css/blog.css
  js/blog.js
  js/md-renderer.js
  data/*.json
  figures/*.svg
  figures/*.png
pages/
  story.md
  math.md
  experiments.md
  audit.md
  reproducibility.md
README.md
```

主页面把最重要的叙事直接写入 HTML，深层内容以 Markdown 阅读层打开。Markdown 内部链接统一转换为带 `data-md-link` 的页面内锚点，避免 modal 关闭后丢失回退位置。所有外部链接使用 `target=_blank` 和 `rel=noopener`。

图表优先使用静态 SVG/PNG，避免依赖运行时数据服务。所有图表保存生成脚本和数据 JSON。手机端隐藏侧栏并启用横向导航，表格允许横向滚动，数学块允许横向滚动。

## 12. 原始材料链接

页面至少提供以下可读入口。

- `PR_PV_MFAPS_DERIVATION_PACKAGE_2026-11-07.md`
- `PR_PV_MFAPS_CLAIMS_EVIDENCE_AUDIT.md`
- `KAPPA_MINIMAX_ALLOCATION.md`
- `MINIMAX_KAPPA_PROPOSITION.md`
- `TAIL_FB_GEO_DERIVATION.md`
- `PAPER_CLAIM_AUDIT_FUSED_2026-11-06.md`
- `PAPER_CLAIM_AUDIT_PIVOT_2026-11-05.md`
- `CROSS_DOMAIN_PLAN.md`
- `ARIS_WORKFLOW_TRACE_2026-11-04.md`

每个链接都要把 Markdown 转成页面内可读格式，确保标题、公式、代码和跳转路径可返回。
