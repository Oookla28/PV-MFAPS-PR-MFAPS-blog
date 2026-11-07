# 实验路线与消融记录

## 1. 机制阶段

E0 在 20 个 Raw/Log endpoint 上固定 encoder，比较当前读出和参考读出。E1 从相同 checkpoint 分叉八步短 horizon，观察反馈差异是否改变后续表示。B3 将 native、two-forward native 和 adaptive reference correction 放进同一训练协议。E2 搜索前向变化很小、encoder gradient 变化较大的方向。

这四组实验把“模型读不出来”拆成了三个可测对象。表示是否包含类别信息，读出是否已经充分拟合，以及读出误差怎样进入 encoder update。机制阶段使用 FIT-only 诊断，HELD 不参与选择。

## 2. 3090 阶段

| 分支 | HELD Macro-F1 | HELD Tail-10% | 实际作用 |
|---|---:|---:|---|
| mean TF-GPS | 低于主要基线 | 低于主要基线 | 暴露类均值调度的方差传播 |
| CVaR-TF-GPS | 0.4486 | 0.2341 | 显示尾部—宏观折中 |
| corrected K-MFAPS | 0.4615 | 0.2778 | 说明尾部保护可以提高，但成本显著增加 |
| cost-aware PC-FA | 0.4680 | 0.2381 | 把 proxy budget 与真实 wall-clock 分开 |
| fused PC-FA | 由 3 次降至 1 次物理 encoder forward | — | 引出系统协同路径 |

这一阶段的作用是确定风险包络必须同时包含收缩、稳定性和 measured cost。单独优化尾部或单独压低 FLOPs 都不足以描述最终目标。

## 3. Balanced Softmax 插件实验

这组结果是自行执行的真实插件矩阵。协议为两个 partition、三个 training seeds、八步 micro-proxy、共享 backbone 和一次性 HELD。冻结前所有 artifact 的 `held_reads=0`，冻结后读取 50,208 个样本结果。

| Arm | HELD Macro-F1 | HELD Tail-10% | 相对 Balanced Softmax 的 Macro | 相对 Balanced Softmax 的 Tail | wall / FLOPs |
|---|---:|---:|---:|---:|---:|
| balanced_softmax | 0.4727 | 0.2659 | — | — | 0.555 s / (1.077\times10^{12}) |
| bs_fixed_k_matched_B02 | 0.4553 | 0.2659 | −1.74 pp | 0.00 pp | 0.630 s / (1.257\times10^{12}) |
| bs_pcfa | 0.4773 | 0.2738 | +0.46 pp | +0.79 pp | 0.571 s / (1.077\times10^{12}) |
| native_plus_pv_mfaps | 0.4653 | 0.2778 | −0.74 pp | +1.19 pp | 0.560 s / (1.077\times10^{12}) |
| effective_number | 0.4760 | 0.2778 | +0.33 pp | +1.19 pp | 0.537 s / (1.077\times10^{12}) |

`bs_pcfa` 说明反馈分配可以作为 Balanced Softmax 的插件工作。`native_plus_pv_mfaps` 单独提高 Tail-10% 而降低 Macro-F1，显示损失函数和反馈策略承担的是不同角色。

## 4. A100 action-space

A100-SXM4-40GB 上完成 36 配置 pilot，selector 选出 3 个 FIT-only 候选，随后完成 final freeze、一次 HELD 和 deterministic audit。A100 阶段把动作空间、measured cost、warm-up 和 CUDA event 计时固定在同一协议中，最终把算法动作与物理执行分开记录。

## 5. 分区 gate 与 epsilon sweep

分区异质性 gate 只使用 FIT。Macro gap 为 0.0139206464，超过预注册的 0.01 阈值。Tail gap 为 0.0029066，action TV 为 0.02899。Macro gap 触发 PR 注册。

| epsilon | 动作直方图 | 相对 epsilon=0.10 的变化 |
|---:|---|---:|
| 0.10 | `{0:61, 2:2, 4:6}` | 基准 |
| 0.05 | `{0:62, 2:1, 4:6}` | 1 类 |
| 0.01 | `{0:61, 1:1, 2:1, 4:6}` | 2 类 |
| 0.001 | `{0:60, 1:3, 4:6}` | 4 类 |
| 0 | `{0:60, 1:3, 4:6}` | 4 类 |

p0 与 p1 的逐类 counts 完全配对，所以 epsilon sweep 中的动作变化来源于风险包络，而不是分区频率差异。

## 6. 最终 clean matched

PR-only 与 clean controls 使用独立目录。最终冻结 24 个 unique states，HELD 一次性读取 25,104 个样本结果。

| Method | HELD Macro-F1 | HELD Tail-10% | 最大 wall ratio | 最大 FLOPs ratio |
|---|---:|---:|---:|---:|
| fixed-k matched | 0.4553 | 0.2659 | 1.0000 | 1.0000 |
| effective-number | 0.4760 | 0.2778 | — | — |
| PR-PV-MFAPS | 0.4841 | 0.2976 | 0.9974 | 0.8889 |

PR 相对 fixed-k 的 Macro-F1 提升 0.0288，Tail-10% 提升 0.0317。相对 effective-number 的比较单独报告，因为 wall ratio 为 1.0768，和 fixed-k 的 1.05× budget gate 属于不同问题。

`bs_pcfa` 在 PR 终链中复用了同一 eps=0.01 policy 和 fused kernel，因此作为 duplicate sanity control 保存，不作为独立基线。
