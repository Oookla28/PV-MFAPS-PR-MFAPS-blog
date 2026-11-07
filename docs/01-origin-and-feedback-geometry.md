# 一开始比较的是几何，后来追问的是反馈

最初的问题很朴素。对同一批植物病害样本，比较 Raw、矩阵 Log 和其他 SPD 映射，看看哪一种几何让分类更好。固定编码器、重新拟合 affine readout 时，Raw 与 Log 都可以被读出来。把编码器解冻后，同一表示的训练轨迹却会分开。

> 表示里有信息，和当前的读出是否会把正确的信号传回编码器，是两个问题。

对样本 $x_i$，记 SPD 表示为 $C_\psi(x_i)$，几何映射为 $\phi$，读出特征为

$$z_\psi(x_i)=\operatorname{svec}(\phi(C_\psi(x_i))).$$

读出 $W$ 和编码器参数 $\psi$ 的联合目标写成

$$L(\psi,W)=\frac1n\sum_i\ell(Wz_\psi(x_i),y_i)+\frac{\lambda}{2}\lVert W\rVert_F^2.$$

在相同表示上充分求解读出，得到 $W^*(\psi)$。训练中实际使用的头是 $W_t$。于是可以直接比较

$$g_{\mathrm{actual}}=\nabla_\psi L(\psi,W_t),\qquad g_{\mathrm{ref}}=\nabla_\psi L(\psi,W^*(\psi)).$$

读出反馈失配定义为

$$e_\psi=g_{\mathrm{actual}}-g_{\mathrm{ref}}.$$

## Loewner 差商把前向和反向分开

令 $C=U\Lambda U^\top$ 是对称正定矩阵。谱函数 $\phi$ 的 Fréchet 导数满足

$$D\phi_C[H]=U\big(G_\phi(\Lambda)\odot(U^\top H U)\big)U^\top,$$

其中

$$\big(G_\phi(\Lambda)\big)_{ij}=\begin{cases}\dfrac{\phi(\lambda_i)-\phi(\lambda_j)}{\lambda_i-\lambda_j},&\lambda_i\ne\lambda_j,\\[6pt]\phi'(\lambda_i),&\lambda_i=\lambda_j.\end{cases}$$

前向特征主要由 $\phi(\lambda_i)$ 决定，反向反馈则被差商 $G_\phi$ 调制。对矩阵 Log，$\phi'(\lambda)=1/\lambda$，小特征值会让导数方向变得敏感。稳定的矩阵 Log 反传、上游编码器 Jacobian 和读出求解精度必须分开测。

## E0、E1、B3 和 E2 如何拆开问题

- **E0** 在 20 个 Raw/Log checkpoint endpoint 上测固定特征、参考读出、梯度方向和失配量。
- **E1** 用短 horizon 分叉编码器，检查反馈差异是否改变后续表示。
- **B3** 比较 native、two-forward native 和 adaptive reference correction。Raw 与 Log 的收益方向不同，出现几何依赖边界。
- **E2** 搜索前向弱方向。若 $ZU$ 很小而 $RU^\top$ 不小，当前 logits 近似不变也不能推出编码器梯度不变。

这些机制诊断主要使用 FIT-only 协议。HELD 没有被当成诊断输入。
