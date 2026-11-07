# PR-PV-MFAPS：分区鲁棒反馈精度分配的数学推导

> 这份章节解释算法如何从有限动作风险包络、KL 分区鲁棒目标和离散动态规划得到。它证明的是注册的优化问题与有限动作求解性质，不把求解证书升级成神经网络总体泛化定理。

## 1. 动作、风险与成本

数据有两个分区 (p\in\{0,1\}) 和 (C) 个类别。每个类别选择一个共享动作

$$
a_c=(k_c,\alpha_c)\in\mathcal A=\mathcal K\times\mathcal G,
\qquad \mathcal K=\{0,1,2,4\}.
$$

其中 (k_c) 是读出内循环步数，(alpha_c) 是 reference gradient mixing coefficient。策略对两个分区共享，这是 PR 约束的关键。若 p0 和 p1 各自求一套动作，就只是在重复 partition-specific allocation，不能回答分区鲁棒问题。

对每个分区、类别和动作，校准风险包络写作

$$
\widehat r_{p,c}(a)=b_{p,c}(a)\rho_{p,c}(a)^{2k}+s_{p,c}(a)+\beta_{p,c}(a)+e_{p,c}(a;\delta).
$$

四项分别表示内层偏差、有限批次统计、条件稳定性和置信半径。每项的数值必须能回到校准工件，不能从 HELD 结果反推。

类别风险与成本按类加和：

$$
R_p(a)=\sum_{c=1}^{C}\widehat r_{p,c}(a_c),
\qquad
C_p(a)=\sum_{c=1}^{C}\widehat q_{p,c}(a_c).
$$

这里的 (q) 可以在 preflight 中使用 proxy，但最终 matched gate 必须使用 measured fixed-k cost。否则动作策略可能在理论 FLOPs 上节省，实际 wall-clock 却更慢。

## 2. KL 歧义集与鲁棒风险

令名义分区混合为 (ar u=(\bar u_0,\bar u_1))，定义

$$
\mathcal U_\varepsilon=\left\{u\in\Delta^2:D_{\mathrm{KL}}(u\Vert\bar u)\le\varepsilon\right\}.
$$

对任意两个分区风险向量 (v=(v_0,v_1))，鲁棒聚合为

$$
\Phi_\varepsilon(v)=\sup_{u\in\mathcal U_\varepsilon}u^\top v.
$$

共享策略的优化问题为

$$
\min_{a_1,\ldots,a_C\in\mathcal A}\Phi_\varepsilon\!\left(R_0(a),R_1(a)\right)
$$

满足 measured budget 约束

$$
\Phi_\varepsilon\!\left(C_0(a),C_1(a)\right)\le B_{\mathrm{measured}}.
$$

若预注册了 macro guard，还需满足

$$
\Phi_\varepsilon\!\left(M_0(a),M_1(a)\right)\le M_{\mathrm{native}}+\delta_M.
$$

macro guard 是否启用必须在读取 HELD 前决定，不能根据 HELD 结果再补上。

## 3. KL 对偶推导

KL 约束的拉格朗日形式为

$$
\sup_{u\in\Delta^2}\left\{u^\top v-\tau\left(D_{\mathrm{KL}}(u\Vert\bar u)-\varepsilon\right)\right\},\qquad \tau>0.
$$

将 KL 展开并对 (u) 做一阶条件，可得到指数倾斜形式

$$
 u_p(\tau)=\frac{\bar u_p\exp(v_p/\tau)}{\sum_{j=0}^{1}\bar u_j\exp(v_j/\tau)}.
$$

代回后得到一维对偶

$$
\Phi_\varepsilon(v)=\inf_{\tau>0}\left\{\tau\varepsilon+\tau\log\!\left(\sum_{p=0}^{1}\bar u_p\exp(v_p/\tau)\right)\right\}.
$$

实现中使用 log-sum-exp，并记录最优 (	au)、KL 残差和边界状态。对于 (arepsilon=0)，取极限得到名义平均

$$
\Phi_0(v)=\bar u_0v_0+\bar u_1v_1.
$$

(Phi_\varepsilon) 对分量单调，并满足无穷范数 Lipschitz 界

$$
\left|\Phi_\varepsilon(v)-\Phi_\varepsilon(w)\right|\le\lVert v-w\rVert_\infty.
$$

因此，若风险向量每个分区的量化误差不超过 (Delta)，鲁棒目标误差不会超过 (Delta) 的最大分量。

## 4. 条件稳定性项

假设类别读出目标为 (lambda)-strongly convex 且 (L)-smooth，梯度下降步长满足 (eta\le1/L)。轨迹收缩率为

$$
\rho=\max\{|1-\eta\lambda|,|1-\eta L|\}<1.
$$

若相邻数据集只替换一个样本，单步梯度扰动不超过 (2G/n_c)，则两条 (k) 步轨迹满足

$$
\begin{aligned}
\lVert W_k(S)-W_k(S')\rVert
&\le \frac{2\eta G}{n_c}\sum_{j=0}^{k-1}\rho^j\\
&=\frac{2\eta G}{n_c(1-\rho)}(1-\rho^k).
\end{aligned}
$$

若外层损失对 (W) 是 (L_\ell)-Lipschitz，可写出

$$
\beta_c(k)\le \frac{2L_\ell\eta G^2}{n_c(1-\rho_c)}(1-\rho_c^k).
$$

这条式子的逻辑边界很重要：在收缩假设下它饱和，不会凭空产生 (gamma^k) 的指数爆炸。若实验中 (k=4) 或 (k=8) 变差，需要从有限样本项、(
ho) 估计误差、任务方向或真实训练轨迹寻找证据。

## 5. 有限动作 Pareto DP

每个动作 (a_c) 有离散成本 (q_c(a_c)) 和两个分区的风险增量

$$
\mathbf r_c(a_c)=\left(\widehat r_{0,c}(a_c),\widehat r_{1,c}(a_c)\right).
$$

将成本离散为整数桶 (b)，将每个分区风险离散为风险索引 (r=(r_0,r_1))。定义前 (c) 个类别的可达状态集合

$$
\mathcal S_c(b,r)=\left\{\text{using first }c\text{ classes, cost }b\text{ and risk index }r\right\}.
$$

转移为

$$
\mathcal S_c(b,r)=\bigcup_{a\in\mathcal A}\mathcal S_{c-1}\!\left(b-q_c(a),\;r-\mathbf r_c(a)\right)\oplus a.
$$

实现中对被支配状态做 Pareto pruning。若状态 (s) 的成本不更低、两个分区风险也不更低，则它不会在任何单调的 (Phi_\varepsilon) 下优于另一个状态，可以安全删除。最终在 (b\le B) 的状态中取

$$
 a^*=\arg\min_{a\in\mathcal S_C}\Phi_\varepsilon\!\left(R_0(a),R_1(a)\right),
$$

并通过 predecessor 指针回溯每个类别的动作。

## 6. 量化残差与证书边界

若风险量化后每个分区的误差满足

$$
\left|R_p(a)-\widetilde R_p(a)\right|\le\Delta_R,
\qquad p\in\{0,1\},
$$

由 (Phi_\varepsilon) 的 Lipschitz 性可得

$$
\left|\Phi_\varepsilon(R(a))-\Phi_\varepsilon(\widetilde R(a))\right|\le\Delta_R.
$$

所以 DP 求解器对注册的有限动作、量化成本问题给出可复核解。它没有证明连续动作空间最优，也没有证明神经网络 population risk 的最优，更没有自动证明跨任务泛化。

## 7. epsilon sweep 的解释

在配对分区中，p0 和 p1 的 69 类 FIT counts 完全相同。实测 (arepsilon) sweep 产生 5 个不同动作向量。最终 (arepsilon=0.01) 相对 (arepsilon=0.1) 只改变两个类别：class 38 从 (k=2) 变为 (k=0)，class 68 从 (k=0) 变为 (k=1)。

这两个动作都沿着 p0 的支配风险面移动。class 38 的 (k=2) 会让 p0 风险上升而只让 p1 小幅下降，因此在更紧的鲁棒目标下被放弃。class 68 的 (k=1) 能降低 p0 风险，即使 p1 略有上升，也会被鲁棒目标选择。

> 这个结果的价值不在于动作改变很多，而在于它证明了动作改变可以在频率被严格配对后仍然发生，来源只能回到分区风险几何。
