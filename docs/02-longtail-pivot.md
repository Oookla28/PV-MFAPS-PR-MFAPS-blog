# 长尾转折：预算应该给谁，以及为什么

PlantSPD 的 FIT 包含 69 个类别，类别计数从 5 到 344，头尾比为 68.8。长尾使“多解几步读出”不再是一个中性的动作。对头部类，它可能只是重复计算；对尾部类，它可能降低当前偏差，也可能让有限样本估计变得更不稳定。

## 1. 第一轮结果如何逼出风险包络

mean TF-GPS 在 FIT 与 HELD 上都下降，说明把所有类别的反馈风险简单平均并不能消除尾部噪声。CVaR-TF-GPS 找到了一点尾部—宏观折中，但仍低于 effective-number。corrected K-MFAPS 在 HELD Tail-10% 上达到 0.2778，却付出了约 (2.154\times10^{12}) FLOPs 的成本。第一轮 cost-aware PC-FA 在 FLOPs 上看似合规，却没有同时通过 wall-clock 和 HELD tail gate。

| 分支 | HELD Macro-F1 | HELD Tail-10% | 读法 |
|---|---:|---:|---|
| effective-number | 0.4760 | 0.2778 | 宏观指标强的基线 |
| fixed-k matched | 0.4553 | 0.2659 | 最终预算参照 |
| corrected K-MFAPS | 0.4615 | 0.2778 | tail trade-off，成本很高 |
| cost-aware PC-FA | 0.4680 | 0.2381 | FIT tail 没有在 HELD 复现 |

> 这次失败不是一个超参数的问题。代理预算、真实 wall-clock 和外推风险被混成一个标量，才是更深的错误。

## 2. 有限动作风险

对类别 (c) 和内循环动作 (k)，写一个明确声明来源的风险代理：

$$
R_c(k)=A_c\rho_c^{2k}+s_c(k)+\beta_c(k)+e_c(\delta).
$$

第一项是读出残差的收缩包络，(s_c(k)) 表示有限批次统计，(eta_c(k)) 是条件稳定性项，(e_c(\delta)) 是置信半径。它们不能被随意合并为一个“复杂度惩罚”。

假设内层读出目标是 (lambda)-strongly convex 且 (L)-smooth，步长满足 (eta\le 1/L)，则

$$
\rho=\max\{|1-\eta\lambda|,|1-\eta L|\}<1.
$$

若替换一个样本最多使一次内层梯度改变 (2G/n_c)，对 (S) 和 (S') 的 (k) 步轨迹进行耦合，得到

$$
\begin{aligned}
\lVert W_k(S)-W_k(S')\rVert
&\le \frac{2\eta G}{n_c}\sum_{j=0}^{k-1}\rho^j\\
&=\frac{2\eta G}{n_c(1-\rho)}(1-\rho^k).
\end{aligned}
$$

若外层损失对读出参数是 (L_\ell)-Lipschitz，则条件稳定性项可写为

$$
\beta_c(k)\le \frac{2L_\ell\eta G^2}{n_c(1-\rho_c)}(1-\rho_c^k).
$$

这个项在假设成立时趋于饱和，不是指数爆炸。实验里观察到的尾部恶化必须由实测 (s_c(k))、估计误差或训练轨迹来解释，不能从稳定性式子直接推出普适方差爆炸。
