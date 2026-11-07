# 从反馈机制转向长尾分配

PlantSPD 的 FIT 有 69 个类别，最少 5 个样本，最多 344 个样本，头尾比为 68.8。把计算全部交给最差类别，Tail-10% 可能上升，Macro-F1 却会下降。把所有类别平均处理，平均梯度方差又会把两头都拖住。

最早的 mean TF-GPS 在 FIT 和 HELD 上都失败。CVaR-TF-GPS 找到了一点尾部—宏观折中，但仍低于 effective-number。corrected K-MFAPS 能提高 HELD tail-10%，代价是大幅增加 wall-clock。第一轮 3090 cost-aware PC-FA 通过了 FLOPs 预算，却没有通过 wall-clock 和 HELD tail gate。

| 分支 | HELD Macro-F1 | HELD Tail-10% | 解释 |
|---|---:|---:|---|
| effective-number | 0.4760 | 0.2778 | 平均指标强，尾部不突出 |
| fixed-k matched | 0.4553 | 0.2659 | 预算对照 |
| corrected K-MFAPS | 0.4615 | 0.2778 | 尾部 trade-off，成本很高 |
| cost-aware PC-FA | 0.4680 | 0.2381 | FIT tail 增益没有在 HELD 重现 |

> 这次失败不是一个超参数的问题，而是代理预算、实际 wall-clock 和外推风险被混成了一个标量。

## 预算反噬的风险包络

为每个类别写一个有限动作风险代理

$$R_c(k)=A_c\rho_c^{2k}+\frac{\sigma_c^2}{n_c}\xi^k.$$

第一项描述内层读出残差，第二项描述一个显式声明的有限样本项。这里的 $\xi^k$ 不能无条件当作稳定性定理。若内层目标是 $\lambda$-strongly convex 且 $L$-smooth，步长 $\eta\le 1/L$，轨迹收缩率为

$$\rho=\max\{|1-\eta\lambda|,|1-\eta L|\}<1,$$

则单样本替换带来的轨迹差异满足

$$\lVert W_k(S)-W_k(S')\rVert\le \frac{2\eta G}{n(1-\rho)}(1-\rho^k).$$

当外层损失对读出是 $L_\ell$-Lipschitz 时，可用

$$\beta_c(k)\le \frac{2L_\ell\eta G^2}{n_c(1-\rho_c)}(1-\rho_c^k)$$

作为条件稳定性项。它在这些假设下趋于饱和，不是指数爆炸。经验上看到的增长只能写成诊断项，不能倒推成普适定理。
