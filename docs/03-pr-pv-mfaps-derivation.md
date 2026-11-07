# PR-PV-MFAPS derivation package (registered-theory draft)

Status: `REGISTERED_NOT_EXECUTED` at creation time. This document defines the
mathematical contract for the partition-robust extension. It does not certify
an empirical gain or a generalisation theorem for the trained network.

## 1. Objects and action space

There are two data partitions, `p in {0,1}`, and `C` classes. For each class
we select one finite action

\[
 a_c=(k_c,\alpha_c)\in {\cal A}={\cal K}\times {\cal G},
 \qquad {\cal K}=\{0,1,2,4\},
\]

where `k` is the number of inner readout steps and `alpha` is the reference
gradient mixing coefficient. The policy is shared by the two partitions; this
is essential. Solving one policy per partition would merely reproduce the
non-robust partition-specific allocation.

For a calibrated partition/class/action triple, let

\[
 \widehat r_{p,c}(a) =
 b_{p,c}(a)\rho_{p,c}(a)^{2k}
 + s_{p,c}(a)
 + \beta_{p,c}(a)
 + e_{p,c}(a;\delta)
\]

be the declared feedback-risk envelope. The first term is the measured inner
solve bias envelope. `s` is the finite-batch variance term. `beta` is an
algorithmic-stability term and `e` is a confidence radius. The code may use a
different certified decomposition, but the final artifact must expose every
term and its source. No term may be silently inferred from the HELD set.

Partition risk and cost are additive over classes:

\[
 R_p(a)=\sum_c \widehat r_{p,c}(a_c),\qquad
 C_p(a)=\sum_c \widehat q_{p,c}(a_c).
\]

`q` is a measured or contract-defined per-class cost. A measured fixed-k cost
must be used for the final system budget; the proxy cost is only a preflight
quantity.

## 2. Partition ambiguity and robust objective

Let `ubar` be the registered nominal partition mixture and let

\[
 {\cal U}_\varepsilon=
 \{u\in\Delta^2:D_{KL}(u\|\bar u)\le\varepsilon\}.
\]

For a two-vector `v=(v_0,v_1)`, define

\[
 \Phi_\varepsilon(v)=\sup_{u\in {\cal U}_\varepsilon}u^\top v.
\]

The PR policy is the shared finite action vector solving

\[
 \min_{a_c\in{\cal A}} \Phi_\varepsilon(R_0(a),R_1(a))
\]

subject to

\[
 \Phi_\varepsilon(C_0(a),C_1(a))\le B_{\rm measured},
\]

and, if the macro-risk guard is enabled,

\[
 \Phi_\varepsilon(M_0(a),M_1(a))\le M_{\rm native}+\delta_M.
\]

The macro guard is optional only if the registration says so. It is not
allowed to be tuned against HELD. The budget is the measured fixed-k budget
multiplied by the registered tolerance (for example `1.05`).

### KL dual

For `epsilon >= 0`,

\[
 \Phi_\varepsilon(v)=
 \inf_{\tau>0}\left\{
 \tau\varepsilon+\tau\log\!\left(
 \sum_{p=0}^1\bar u_p e^{v_p/\tau}\right)\right\}.
\]

The `epsilon=0` limit is `ubar^T v`. The implementation must evaluate this
with a log-sum-exp operation and a one-dimensional bounded minimisation. A
direct grid over `tau` is acceptable for a finite preflight, but the selected
value and residual must be recorded.

The map `Phi` is monotone and 1-Lipschitz in the infinity norm:

\[
 |\Phi_\varepsilon(v)-\Phi_\varepsilon(w)|
 \le \|v-w\|_\infty.
\]

This gives a useful audit bound when risk values are quantised.

## 3. Stability and confidence envelope

The previous informal claim that a variance term must grow as `gamma^k` is not
valid without an explicit unstable-update assumption. The registered theory
uses a standard conditional statement instead.

Assume the class/readout empirical objective is `lambda`-strongly convex and
`L`-smooth, and inner gradient descent uses `eta <= 1/L`. Let

\[
 \rho=\max\{|1-\eta\lambda|,|1-\eta L|\}<1,
\]

and suppose a single-example replacement changes one inner gradient by at
most `2G/n`. Coupling the two k-step trajectories gives

\[
 \|W_k(S)-W_k(S')\|
 \le {2\eta G\over n}\sum_{j=0}^{k-1}\rho^j
 \le {2\eta G\over n(1-\rho)}(1-\rho^k).
\]

If the outer loss is `L_l`-Lipschitz in the readout, the uniform-stability
contribution is bounded by

\[
 \beta_c(k)\le {2L_l\eta G^2\over n_c(1-\rho_c)}(1-\rho_c^k).
\]

This term saturates under the stated assumptions; it does **not** explode
exponentially. If the implementation observes an increasing empirical term,
that is a diagnostic, not a theorem, and must be reported as such.

For a bounded loss in `[0,1]`, a simple confidence radius is

\[
 e_{p,c}(a;\delta)=
 2G_{p,c}(a)\sqrt{\frac{\log(2PC|{\cal A}|/\delta)}{n_{p,c}}}
 +\frac{3\log(2PC|{\cal A}|/\delta)}{n_{p,c}}.
\]

The radius is a simultaneous finite-action bound. It is valid only if the
calibration samples are independent of the final HELD evaluation and the
bounded-gradient assumptions are checked or explicitly declared.

Consequently, with probability at least `1-delta`, every fixed finite policy
has population partition risk no larger than the calibrated envelope plus the
corresponding confidence terms. Since `Phi` is monotone, the robust population
objective is bounded by applying `Phi` to the per-partition upper envelopes.
This is a conditional finite-sample statement, not a claim that the neural
encoder is globally PAC-learned.

## 4. Exact finite-action DP and quantisation error

The robust objective is not separable because `Phi` is applied after summing
class risks. A scalar DP that stores only a single risk value is therefore not
exact. The registered solver uses a Pareto-frontier DP.

For class prefix `j`, store states

\[
 (b,r_0,r_1,c_0,c_1,\pi),
\]

where `b` is quantised cost, `r_p` is quantised accumulated risk,
`c_p` is quantised partition cost, and `pi` is the predecessor/action map.
For each action `a` of class `j+1`, append the state and discard a state if
another state has no larger budget, no larger `(r_0,r_1)`, and no larger
`(c_0,c_1)` with at least one strict inequality. At the final layer retain
states satisfying the robust cost and macro guard and select the minimum
`Phi_epsilon((r_0,r_1))`.

If risks are quantised with step `Delta_r` and costs with step `Delta_c`, the
returned policy is feasible for the unquantised budget after adding the
registered safety margin `C*Delta_c`; its robust-risk suboptimality is at most
`P*C*Delta_r` because `Phi` is 1-Lipschitz and there are `C` class additions.
The certificate must report frontier size, grid steps, discarded-state count,
and the final quantisation residual.

For large class counts the frontier can grow exponentially. In that case an
approximate frontier is allowed only as a separately named variant; it must
report the dominance tolerance and may not be called an exact DP.

## 5. What can and cannot be claimed

### Claimable after deterministic preflight only

* the KL dual and DP solver satisfy the registered numerical contracts;
* the policy is shared across partitions and respects the declared finite action
  set;
* the quantisation and budget residuals are within their registered bounds.

### Requires FIT matched execution

* PR-PV-MFAPS actually produces the registered action sequence;
* measured FLOPs/wall time satisfy the budget comparison;
* the partition-robust objective is lower than its registered controls on FIT.

### Requires one-time HELD and claims audit

* any held-out macro or tail improvement;
* any statement that the robust policy improves generalisation;
* any SOTA, cross-domain, or universal-improvement statement.

No single successful gate upgrades an unexecuted transfer experiment. The
partition gate that triggers registration is FIT-only and therefore cannot be
used as evidence for HELD efficacy.

