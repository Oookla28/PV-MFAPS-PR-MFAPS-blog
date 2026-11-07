# 反馈几何与分区鲁棒内循环分配

## PlantSPD 长尾植物病害识别中的机制、算法与系统协同

## 摘要

植物病害识别首先面对的是一个工程问题：在极端长尾类别分布下，如何让模型识别稀有病害。双模型 C0+S2、有效重采样和长尾损失可以改善训练过程，但它们没有解释一个反复出现的现象。模型在 FIT 上已经能够读出类别信息，HELD 上的尾部类别却明显下降。本文将问题从采样频率转向读出反馈。

对 SPD 协方差表示，谱函数决定前向坐标，Loewner 差商决定局部扰动如何传回编码器。当读出头只经过有限步优化时，当前读出产生的 encoder gradient 与同一表示上的充分拟合读出产生的参考 gradient 可能不同。本文将两者之差定义为反馈失配，并以此构造类别级的内循环风险包络。风险包络同时包含读出残差收缩、有限批次统计、条件稳定性和置信半径。随后，有限动作集合上的离散动态规划在 measured budget 约束下分配每个类别的内循环精度。

两个数据分区的 FIT counts 逐类配对相同，分区异质性只能由风险几何解释。针对这一结构，进一步使用 KL 歧义集构造共享的分区鲁棒策略 PR-PV-MFAPS。最终 PlantSPD matched 结果中，PR-PV-MFAPS 的 HELD Macro-F1 为 0.4841，Tail-10% 为 0.2976。相对 fixed-k，分别提高 0.0288 和 0.0317，最大 wall-clock ratio 为 0.9974，最大 FLOPs ratio 为 0.8889。独立的 Balanced Softmax 插件实验在两个分区和三个训练种子上得到 `bs_pcfa` 的 HELD Macro-F1 0.4773 和 Tail-10% 0.2738。

在系统层面，Fused 执行路径将真实 checkpoint 上的物理 encoder forward 从 3 次降到 1 次。legacy 与 fused 的动作序列和逻辑计数一致，最大 encoder state absolute delta 为 (5.772919\times10^{-6})，CUDA event time 从 0.602 s 降至 0.157 s。研究由此形成一条完整路径，从病害检测和 SPD 表示出发，经由反馈机制、风险分配和分区鲁棒优化，最后落到可验证的 GPU 执行图。

## 1. 研究问题

PlantSPD 包含 69 个植物病害类别。FIT 有 4,184 个样本，HELD 有 1,046 个样本，FIT 类别计数范围为 5 到 344，头尾比为 68.8。这样的数据分布使宏观指标和最差尾部指标具有不同的统计含义。

研究起点并不是一个新的优化器。最初的目标是通过 C0+S2 双模型或有效重采样改善病害检测。工程链条能够运行之后，仍然存在一个更具体的训练问题：

> 为什么一个表示在固定读出下已经可以分类，联合训练时却把不同的梯度信号传回编码器？

这个问题要求把以下对象分开：

1. 表示是否包含标签信息。
2. 当前读出是否已经充分拟合。
3. 当前读出和参考读出产生的 encoder gradient 是否一致。
4. 对反馈进行修正所需要的实际计算是否可接受。

研究采用 FIT-only 校准和训练，所有动作策略在 freeze 前确定。HELD 只在全部 FIT states 冻结之后读取一次。

## 2. SPD 表示与反馈失配

对样本 (x_i)，令编码器输出 SPD 矩阵 (C_\psi(x_i))，谱映射为 (phi)，对称矩阵向量化算子为 (operatorname{svec})。读出特征写作

$$
z_\psi(x_i)=\operatorname{svec}\!\left(\phi(C_\psi(x_i))\right).
$$

令读出矩阵为 (W)，联合训练目标为

$$
L(\psi,W)=\frac1n\sum_{i=1}^{n}\ell\!\left(Wz_\psi(x_i),y_i\right)+\frac{\lambda}{2}\lVert W\rVert_F^2.
$$

固定编码器参数 (psi) 后，充分拟合的参考读出为

$$
W^*(\psi)=\arg\min_W L(\psi,W).
$$

实际训练只对读出执行有限步，因此使用的是 (W_t)。当前读出和参考读出对编码器的反馈分别为

$$
 g_{\mathrm{actual}}=\nabla_\psi L(\psi,W_t),
 \qquad
 g_{\mathrm{ref}}=\nabla_\psi L\!\left(\psi,W^*(\psi)\right).
$$

定义反馈失配

$$
 e_\psi=g_{\mathrm{actual}}-g_{\mathrm{ref}}.
$$

反馈失配不是 readout loss gap 的另一个记号。它经过 (z_\psi) 对 (psi) 的 Jacobian 投影，直接决定 encoder 下一步的更新方向。

### 2.1 前向近似等价与反向差异

将当前 batch 的读出特征堆叠为 (Z)，logits 为 (S=ZW)。若存在头部扰动 (U) 满足

$$
ZU=0,
$$

则 (W) 和 (W+U) 在当前 batch 上给出相同 logits。令 (R=\partial L/\partial S)，两种读出对特征的梯度分别为

$$
\nabla_ZL=RW^\top,
\qquad
\nabla_ZL'=R(W+U)^\top.
$$

因此

$$
\Delta\nabla_ZL=RU^\top.
$$

这个构造说明，前向预测的接近不能推出反馈的接近。读出是否适合继续训练 encoder，需要单独测量 gradient alignment、gradient norm ratio 和后续 representation drift。

### 2.2 Loewner 差商

设 (C=U\Lambda U^\top) 为对称正定矩阵。对谱函数 (phi)，Fréchet 导数满足

$$
D\phi_C[H]
=U\!\left(G_\phi(\Lambda)\odot(U^\top H U)\right)U^\top.
$$

Loewner 差商矩阵为

$$
(G_\phi)_{ij}=\begin{cases}
\dfrac{\phi(\lambda_i)-\phi(\lambda_j)}{\lambda_i-\lambda_j},&\lambda_i\ne\lambda_j,\\[9pt]
\phi'(\lambda_i),&\lambda_i=\lambda_j.
\end{cases}
$$

前向表示主要由 (phi(\lambda_i)) 决定，局部反馈则由 (G_\phi) 调制。对矩阵 Log，(phi(\lambda)=\log\lambda)，有

$$
\phi'(\lambda)=\frac1\lambda.
$$

当特征值接近零时，局部谱方向会更敏感。数值实现中的近重根反向传播、谱映射本身的敏感性和读出未充分收敛造成的反馈误差属于不同层次，必须分开处理。

### 2.3 机制实验

E0 在 20 个 Raw/Log checkpoint endpoint 上固定 encoder，比较当前读出和参考读出。E1 从相同 checkpoint 分叉八步短 horizon，测量不同反馈对后续 encoder state 的影响。B3 比较 native、two-forward native 和 adaptive reference correction。E2 寻找前向变化很小、encoder gradient 变化显著的方向。

在 E2 的一个 Log 方向检查中，logit delta (L_2) 约为 (8.07\times10^{-4})，encoder gradient delta (L_2) 约为 5.76，梯度差异比约为 3.25。该结果用于说明前向和反向可辨识性之间存在可测分离，不能替换成普适泛化结论。

## 3. 长尾内循环风险包络

长尾训练中，每个类别的读出误差、统计方差和可用计算预算不同。给类别 (c) 分配 (k) 个内循环步，定义有限动作风险代理

$$
R_c(k)=A_c\rho_c^{2k}+s_c(k)+\beta_c(k)+e_c(\delta).
$$

其中：

- (A_c\rho_c^{2k}) 表示读出残差的收缩包络。
- (s_c(k)) 表示有限批次统计项。
- (eta_c(k)) 表示在给定假设下的算法稳定性项。
- (e_c(\delta)) 表示置信半径。

### 3.1 条件稳定性项

假设类别读出目标是 (lambda)-strongly convex 且 (L)-smooth，内层梯度下降步长满足 (eta\le 1/L)。轨迹收缩率写作

$$
\rho=\max\{|1-\eta\lambda|,|1-\eta L|\}<1.
$$

若替换一个类别样本最多使单步内层梯度改变 (2G/n_c)，对两个相邻数据集 (S) 和 (S') 的轨迹进行耦合，得到

$$
\begin{aligned}
\lVert W_k(S)-W_k(S')\rVert
&\le \frac{2\eta G}{n_c}\sum_{j=0}^{k-1}\rho^j\\
&=\frac{2\eta G}{n_c(1-\rho)}(1-\rho^k).
\end{aligned}
$$

若外层损失对读出参数是 (L_\ell)-Lipschitz，则

$$
\beta_c(k)\le \frac{2L_\ell\eta G^2}{n_c(1-\rho_c)}(1-\rho_c^k).
$$

在这些假设下，稳定性项趋于饱和。实验校准中的增长曲线属于经验统计代理，不能从该式直接推出无条件的指数方差爆炸。

### 3.2 偏差停滞

两个分区的内层收缩率中位数分别为

$$
\operatorname{median}(\rho_{p0})=0.9998727333,
\qquad
\operatorname{median}(\rho_{p1})=0.9998982112.
$$

当 (ho) 接近 1 时，增加 (k) 对 (A_c\rho_c^{2k}) 的削减非常有限。FIT calibration 记录的 (k=16) 偏差收益约为 0.405%，而稳定性和置信经验代理增长约为 17.3%。因此，若动作策略在若干类别上选择 (k=0)，它表示边际偏差收益已经低于额外风险和 measured cost，而不是表示内循环求解在理论上没有作用。

## 4. 从类别动作到 PV-MFAPS

定义动作集合

$$
\mathcal A=\mathcal K\times\mathcal G,
\qquad
\mathcal K=\{0,1,2,4\},
$$

其中 (k_c\in\mathcal K) 是类别 (c) 的读出内循环步数，(alpha_c\in\mathcal G) 是 reference gradient mixing coefficient。

对分区 (p\in\{0,1\})，类别风险和成本分别写作

$$
R_p(a)=\sum_{c=1}^{C}\widehat r_{p,c}(a_c),
\qquad
C_p(a)=\sum_{c=1}^{C}\widehat q_{p,c}(a_c).
$$

基础 PV-MFAPS 在 measured budget 约束下优化风险包络。有限动作 DP 以累计成本和累计风险向量为状态，对被其他状态同时支配的候选进行 Pareto pruning，再通过 predecessor 指针回溯每个类别的动作。若风险量化误差满足

$$
\left\|R(a)-\widetilde R(a)\right\|_\infty\le\Delta_R,
$$

则量化误差对应的目标误差受 (Delta_R) 控制。

## 5. Balanced Softmax 插件实验

这一组实验由当前研究自行执行，用来观察反馈精度分配能否接入一个明确的长尾损失。实验使用两个分区和三个 training seeds，统一 backbone、训练步数、batch、状态路径和成本记录。总计 48 个 matched arms，FIT freeze 后一次性读取 50,208 个 HELD 样本结果。

| Arm | HELD Macro-F1 | HELD Tail-10% | 相对 Balanced Softmax 的 Macro | 相对 Balanced Softmax 的 Tail | wall-clock | FLOPs |
|---|---:|---:|---:|---:|---:|---:|
| balanced_softmax | 0.4727 | 0.2659 | — | — | 0.555 s | (1.077\times10^{12}) |
| bs_fixed_k_matched_B02 | 0.4553 | 0.2659 | −1.74 pp | 0.00 pp | 0.630 s | (1.257\times10^{12}) |
| bs_pcfa | **0.4773** | **0.2738** | **+0.46 pp** | **+0.79 pp** | 0.571 s | (1.077\times10^{12}) |
| native_plus_pv_mfaps | 0.4653 | 0.2778 | −0.74 pp | +1.19 pp | 0.560 s | (1.077\times10^{12}) |
| effective_number | 0.4760 | 0.2778 | +0.33 pp | +1.19 pp | 0.537 s | (1.077\times10^{12}) |

`bs_pcfa` 的两项 HELD 指标都高于 Balanced Softmax。`native_plus_pv_mfaps` 的尾部指标更高而宏观指标下降，说明 long-tail loss 与反馈动作分配对应不同的优化作用。该插件矩阵是本研究自行执行的 micro-proxy evidence，训练步数为 8，不能替换成跨任务长周期训练结果。

## 6. 分区鲁棒策略 PR-PV-MFAPS

p0 和 p1 的类别频率逐类完全配对。分区异质性 gate 的 Macro gap 为 0.0139206464，高于预注册阈值 0.01。Tail gap 为 0.0029066，action total variation 为 0.0289855。Macro gap 触发共享分区鲁棒策略的注册。

令名义分区混合为 (ar u)，KL 歧义集为

$$
\mathcal U_\varepsilon
=\left\{u\in\Delta^2:D_{\mathrm{KL}}(u\Vert\bar u)\le\varepsilon\right\}.
$$

对分区风险向量 (v=(R_0(a),R_1(a)))，定义

$$
\Phi_\varepsilon(v)=\sup_{u\in\mathcal U_\varepsilon}u^\top v.
$$

KL 对偶为

$$
\Phi_\varepsilon(v)
=\inf_{\tau>0}\left\{
\tau\varepsilon+\tau\log\!\left(
\sum_{p=0}^{1}\bar u_p\exp(v_p/\tau)
\right)\right\}.
$$

共享策略求解

$$
\min_{a_1,\ldots,a_C\in\mathcal A}
\Phi_\varepsilon\!\left(R_0(a),R_1(a)\right)
$$

并满足

$$
\Phi_\varepsilon\!\left(C_0(a),C_1(a)\right)
\le B_{\mathrm{measured}}.
$$

(Phi_\varepsilon) 满足无穷范数 Lipschitz 界

$$
\left|\Phi_\varepsilon(v)-\Phi_\varepsilon(w)\right|
\le\lVert v-w\rVert_\infty.
$$

因此，分区风险的量化误差不会被鲁棒聚合放大超过最大分区误差。

### 6.1 两个边界类别

epsilon sweep 的 11 个取值产生 5 个不同动作向量。最终选择 (arepsilon=0.01)，相对 (arepsilon=0.10) 只改变两个类别。

- Class 38 的 FIT 计数为 108，频率排名为 11/69。其 p0 风险从 (61.5807) 随 (k=0) 到 (61.9370) 随 (k=2) 增长，p1 风险却从 (14.1952) 降到 (14.0233)。鲁棒目标由 p0 控制，因此动作由 (k=2) 变为 (k=0)。
- Class 68 的 FIT 计数为 12，频率排名为 60/69。其 p0 风险从 (461.7041) 降到 (461.5838) 在 (k=1)，p1 风险从 (252.2245) 升到 (252.2970)。鲁棒目标仍由 p0 控制，因此动作由 (k=0) 变为 (k=1)。

两个类别的频率在两个分区中完全相同。动作翻转来源于分区风险曲率，而不是重新采样。

## 7. Fused 执行路径

类别动作策略如果逐组触发 encoder forward、reference probe 和 H2D 传输，会产生大量物理执行开销。Fused 路径将相同动作的类别分组，并在一个 encoder graph 中完成 native 与 reference gradient 的组合。

核心执行结构为

```python
k_disc = np.asarray(policy["k_discrete"], dtype=int)
alpha_disc = np.asarray(policy["alpha_discrete"], dtype=float)
action_groups = build_action_groups(k_disc, alpha_disc, batch_labels)
loss_native, loss_reference = fused_pcfa_gradient(
    model=model,
    head=head,
    batch=batch,
    action_groups=action_groups,
)
```

真实 checkpoint 的 legacy/fused one-step equivalence 满足

$$
\operatorname{actions}_{\mathrm{legacy}}
=\operatorname{actions}_{\mathrm{fused}},
$$

$$
\operatorname{counters}_{\mathrm{legacy}}
=\operatorname{counters}_{\mathrm{fused}}.
$$

最大 encoder state absolute delta 为

$$
\max\left|\Delta\theta_{\mathrm{encoder}}\right|
=5.772919\times10^{-6}<10^{-5}.
$$

CUDA event time 从 0.602 s 降至 0.157 s，speedup 为 3.8404。最终 matched run 中，最大 FLOPs ratio 为 0.8889，最大 end-to-end wall ratio 为 1.0379。系统结果描述执行路径，算法结果由独立 HELD 读取决定。

## 8. 最终 PlantSPD matched 结果

A100 终链在 freeze 前识别到内部 state tag 命名碰撞。相关 artifact 被物理隔离，随后采用独立目录完成重跑。最终保留 24 个独立 states，所有 FIT artifact 的 `held_reads=0`。冻结之后唯一一次 HELD 读取包含 25,104 个样本访问。

| Method | HELD Macro-F1 | HELD Tail-10% | Wall-clock | FLOPs |
|---|---:|---:|---:|---:|
| fixed-k matched | 0.4553 | 0.2659 | 0.663 s | (1.257\times10^{12}) |
| effective-number | 0.4760 | 0.2778 | 0.563 s | (1.077\times10^{12}) |
| PR-PV-MFAPS | **0.4841** | **0.2976** | 0.606 s | (1.077\times10^{12}) |

相对 fixed-k，PR-PV-MFAPS 的 HELD Macro-F1 提升

$$
0.4841-0.4553=0.0288,
$$

Tail-10% 提升

$$
0.2976-0.2659=0.0317.
$$

最大 wall-clock ratio 为 0.9974，最大 FLOPs ratio 为 0.8889。相对 effective-number，PR 的 Macro-F1 提升 0.0080，Tail-10% 提升 0.0198，但 wall ratio 为 1.0768，因此该比较和 fixed-k matched gate 分开解释。

## 9. 证据状态

### 已执行

- Raw/Log 反馈机制诊断 E0、E1、B3、E2。
- 3090 阶段的 long-tail allocation、cost-aware PC-FA 和 Fused 相关实验。
- A100 preflight、36 配置 action-space pilot、selector、freeze、HELD 和 deterministic audit。
- Balanced Softmax 插件矩阵。
- PR 数学 smoke、epsilon sweep、离散策略、clean matched、24-state freeze 和一次性 HELD。

### 注册未执行

- CIFAR-LT、CIFAR100-LT 和 ImageNet-LT 跨任务验证。
- Learnable spectral geometry 或 Power(p) 对照。
- 第二个独立 SPD 项目的 feedback-transfer 实验。
- 独立语义 reviewer。

### 待验证

- 逐样本 McNemar 检验。
- Community SOTA 和跨任务泛化。
- Alpha-aware 连续动作最优性。
- Shared-cost PR 的 robust-cost theorem。
- 神经网络总体 population generalization theorem。

## 10. 结论

PlantSPD 的长尾训练问题可以被重新表述为反馈精度分配问题。SPD 谱映射改变的不只是前向坐标，也改变了读出误差通过 Loewner 差商投影回 encoder 的方式。有限内循环在 (ho) 接近 1 的条件下表现出偏差停滞，风险包络因此需要同时记录收缩收益、稳定性和 measured cost。

PV-MFAPS 将类别动作选择转化为有限动作风险优化，PR-PV-MFAPS 进一步把两个配对分区放入同一 KL 鲁棒目标。Balanced Softmax 插件实验显示，这种反馈精度分配可以接入长尾损失。Fused 执行路径则证明了相同逻辑动作可以通过更少的物理 encoder work 完成。

当前证据支持的研究陈述是：在 PlantSPD 的配对分区和 measured fixed-k budget 下，分区鲁棒的反馈精度分配建立了更高的 Macro–Tail measured Pareto point，并且可以由 Fused execution path 在相近 wall-clock 下实现。跨任务、总体泛化和逐样本显著性仍由后续检验决定。
