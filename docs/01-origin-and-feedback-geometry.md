# 反馈几何：从“能读出”到“能正确反馈”

## 1. 最初的比较为什么不够

PlantSPD 的最初问题是比较 Raw、矩阵 Log 以及其他 SPD 映射。固定编码器后，给定同一批 SPD 特征，重新拟合一个 affine readout，Raw 与 Log 都可以被读出。这说明两个表示之间的差异不能直接归结为“一个含有标签信息，另一个没有”。

> 固定特征上的可读性，是一个前向问题。联合训练中的反馈方向，是另一个问题。

对样本 (x_i)，令编码器输出的 SPD 矩阵为 (C_\psi(x_i))，谱映射为 (phi)，对称矩阵向量化算子为 (operatorname{svec})。读出特征为

$$
z_\psi(x_i)=\operatorname{svec}\!\left(\phi(C_\psi(x_i))\right).
$$

读出矩阵为 (W)，联合目标为

$$
L(\psi,W)=\frac1n\sum_{i=1}^{n}\ell\!\left(Wz_\psi(x_i),y_i\right)+\frac{\lambda}{2}\lVert W\rVert_F^2.
$$

在固定 (psi) 时，充分求解读出得到

$$
W^*(\psi)=\arg\min_W L(\psi,W).
$$

训练中真正使用的是有限步读出 (W_t)。于是可以把两个 encoder gradient 放在同一个 checkpoint 上比较：

$$
 g_{\mathrm{actual}}=\nabla_\psi L(\psi,W_t),\qquad
 g_{\mathrm{ref}}=\nabla_\psi L\!\left(\psi,W^*(\psi)\right).
$$

反馈失配定义为

$$
 e_\psi=g_{\mathrm{actual}}-g_{\mathrm{ref}}.
$$

这里的 (e_\psi) 不是读出 loss 的简单差值。它是读出误差经过表示 Jacobian 后的结果，因此同样的前向损失差异可能产生完全不同的 encoder update。

## 2. 一个前向等价、反向不等价的构造

设一批样本的读出特征堆成矩阵 (Z)，logits 为 (S=ZW)。若存在头部扰动 (U) 满足

$$
ZU=0,
$$

则 (W) 与 (W+U) 在当前批次产生相同的 logits。令 (R=\partial L/\partial S)，特征梯度分别为

$$
\nabla_ZL=RW^\top,
$$

以及

$$
\nabla_ZL'=R(W+U)^\top.
$$

因此

$$
\Delta\nabla_ZL=RU^\top.
$$

这个构造排除了一个常见误解：预测不变不代表学习信号不变。它也说明，评估一个读出不能只看当前 accuracy 或 head loss，还要测它对 encoder 的反向作用。

## 3. Loewner 差商怎样进入反馈

令 (C=U\Lambda U^\top) 为对称正定矩阵，(Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_d))。对谱函数 (phi)，Fréchet 导数可写成

$$
D\phi_C[H]=U\!\left(G_\phi(\Lambda)\odot(U^\top H U)\right)U^\top.
$$

Loewner 差商矩阵的元素为

$$
(G_\phi)_{ij}=\begin{cases}
\dfrac{\phi(\lambda_i)-\phi(\lambda_j)}{\lambda_i-\lambda_j},&\lambda_i\ne\lambda_j,\\[9pt]
\phi'(\lambda_i),&\lambda_i=\lambda_j.
\end{cases}
$$

所以前向几何和反向几何由不同对象控制。前向特征主要依赖 (phi(\lambda_i))，而局部梯度扰动经过 (G_\phi) 的逐元素调制。矩阵 Log 满足

$$
\phi(\lambda)=\log\lambda,\qquad \phi'(\lambda)=\frac1\lambda.
$$

小特征值会让某些局部方向更加敏感。但必须将以下三件事分开验证：

1. 稳定 Log 反传是否正确实现了近重根的连续极限；
2. (G_\phi) 如何调制上游扰动；
3. 有限步读出误差如何通过这个 Jacobian 进入 encoder。

## 4. E0、E1、B3、E2 的分工

- **E0**：在 20 个 Raw/Log endpoint 上固定 encoder，分别求当前读出与参考读出，记录梯度范数、余弦相似度和 (e_\psi)。
- **E1**：从同一个 checkpoint 分叉短 horizon，比较当前读出与参考读出对后续 encoder 表示的影响。
- **B3**：在相同训练预算下比较 native、two-forward native 和 reference correction，检查收益是否随几何改变。
- **E2**：寻找满足 (lVert ZU\rVert) 较小而 (lVert RU^\top\rVert) 较大的方向，检验前向弱方向是否真的会改变反馈。

这些实验主要使用 FIT-only 协议。它们的作用是识别机制，不是把短 horizon 结果升级成 HELD 泛化结论。
