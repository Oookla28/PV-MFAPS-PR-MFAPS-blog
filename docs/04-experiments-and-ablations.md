# 实验路线、消融与反转

## 原始机制实验

20 个 Raw/Log endpoint 完成 E0 诊断。E1 做八步短 horizon 分叉。B3 将 native、two-forward native 和 adaptive reference correction 放在同一训练协议中。E2 用 forward-weak/backward-sensitive 方向做反事实检查。

这些实验把“读出没有学好”从一个分数差，拆成可观测的反馈差异。它们没有被用来宣称 Log 优越，也没有把 FIT-only 的短 horizon 结果升级成 HELD 泛化。

## 3090 轮次

| 版本 | 结果 | 状态 |
|---|---|---|
| mean TF-GPS | Macro 与 Tail 同时下降 | 已执行，失败主假设 |
| CVaR-TF-GPS | 尾部略有改善，宏观仍低 | 已执行，trade-off |
| corrected K-MFAPS | Tail-10% 0.2778，wall 约 2.735 s | 已执行，高成本 |
| cost-aware PC-FA | FLOPs 合规，wall 和 HELD tail 不合规 | 已执行，boundary |
| fused PC-FA | 物理 encoder forward 由 3 降到 1 | 已执行，系统优化 |

## A100 action-space

A100-SXM4-40GB 上完成 36 配置 pilot，selector 选出 3 个 FIT-only 候选，随后完成 final freeze、一次 HELD 和 deterministic audit。原 action-space audit 中出现过一个 wall ratio 1.1913 的配置，因此没有把 A100 action-space直接包装成普适算法胜利。

## 分区 gate

分区异质性 gate 只使用 FIT。Macro gap 为 0.0139206464，超过预注册的 0.01 阈值。Tail gap 为 0.0029066，action TV 为 0.02899。这个 gate 触发了 PR 注册，注册本身不等于 HELD 效果。

## epsilon 消融

| epsilon | 动作直方图 | 变化 |
|---:|---|---:|
| 0.10 | `{0:61, 2:2, 4:6}` | 基准选择 |
| 0.05 | `{0:62, 2:1, 4:6}` | 改变 1 类 |
| 0.01 | `{0:61, 1:1, 2:1, 4:6}` | 改变 2 类，最终选择 |
| 0.001 | `{0:60, 1:3, 4:6}` | 改变 4 类 |
| 0 | `{0:60, 1:3, 4:6}` | nominal mean |

## 最终 clean matched

最终 PR run 被拆成 PR-only 和 clean controls 两个目录，避免内部 `bs_pcfa` tag 造成 state collision。合并后有 24 个独立 artifacts 和 24 个 unique states。一次性 HELD 读取 24 个 states，25,104 次样本访问，0 errors。

最终相对 fixed-k 的 gate 通过。PR HELD Macro-F1 为 0.4841，fixed-k 为 0.4553。PR HELD Tail-10% 为 0.2976，fixed-k 为 0.2659。最大 wall 比为 0.9974，最大 FLOPs 比为 0.8889。

相对 effective-number，PR 的 Macro-F1 高 0.0080，Tail-10% 高 0.0198，但 wall ratio 为 1.0768。因此这个比较单独报告，不放进 fixed-k 的 1.05× budget gate。

`bs_pcfa` 使用与 PR 相同的 eps=0.01 policy 和 fused kernel，是 duplicate sanity control，不是独立 baseline。
