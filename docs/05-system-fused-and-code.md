# Fused 系统路径与代码中的关键转折

算法是否有意义，和它是否被正确执行，是两条线。Fused 优化只改执行图，不改动作序列和逻辑计数。

## 等价性约束

真实 checkpoint 的 legacy/fused one-step pair 满足

$$\text{action sequence}_{legacy}=\text{action sequence}_{fused},$$

$$\text{logical counters}_{legacy}=\text{logical counters}_{fused},$$

$$\max |\Delta \theta_{encoder}|=4.158006\times10^{-6}<10^{-5}.$$

物理 encoder forward 从 3 次变成 1 次。matched training 中，8 个训练步包含 20 个逻辑 native pass，但 fused 只执行 8 个物理 encoder forward。

## 代码中的关键结构

```python
k_disc = np.asarray(policy["k_discrete"], dtype=int)
alpha_disc = np.asarray(policy["alpha_discrete"], dtype=float)
active_groups = build_action_groups(k_disc, alpha_disc, batch_labels)
loss_native, loss_reference = fused_pcfa_gradient(
    model=model,
    head=head,
    batch=batch,
    action_groups=active_groups,
)
```

实际代码还处理 warm-up、CUDA event 计时、TF32/cuDNN 确定性、reference probe 计费和 state SHA。这里展示的是结构，完整实现保留在原始远程代码镜像中。

在同一 arm-specific warm-up 协议下，fused PC-FA 相对 fixed-k 的最大 FLOPs 比为 0.8889，平均 0.8593。最大 end-to-end wall ratio 为 1.0379，平均 0.9920。它支持执行路径优化的结论，不能单独推出 tail gain。
