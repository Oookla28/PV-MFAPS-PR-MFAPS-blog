# Blog evidence inventory

This inventory is for the PV-MFAPS / PR-PV-MFAPS research narrative. It preserves the distinction between executed evidence, registered work that has not run, and results that remain unverified.

## Narrative spine

1. **The original PlantSPD question**
   - Fixed-feature readout can be nearly perfect while finite-budget joint fitting behaves differently.
   - Raw and Log are invertibly related representations, so the interesting object is the learning feedback sent to the encoder.
   - Source: `feedback_geometry_pipeline_2026-11-04/NARRATIVE_REPORT.md`, `blog-site/Plant_SPD/00_paper_story/PAPER_STORY_ZH.md`, `blog-site/Plant_SPD/00_paper_story/MATHEMATICS_AND_EVIDENCE.md`.

2. **Mechanism diagnostics**
   - E0 runs cover 20 Raw/Log endpoints. The registered measurements are same-feature probe/readout comparison, gradient norm ratios, cosine alignment and kappa-style mismatch.
   - E1 short-horizon forks and B3 adaptive corrections establish a geometry-dependent boundary. They do not establish a universal algorithmic gain.
   - E2 tests forward-weak directions with backward-sensitive encoder gradients.
   - Source: `feedback_geometry_pipeline_2026-11-04/NARRATIVE_REPORT.md`, `TFGPS_FINAL_AUDIT.md`, local PlantGPU artifacts.

3. **Long-tail scheduling pivot**
   - Mean TF-GPS and early K-MFAPS/PC-FA variants expose a macro-versus-tail conflict and a difference between proxy cost and measured wall-clock cost.
   - Corrected K-MFAPS improves tail metrics in some runs at much higher measured cost.
   - Cost-aware PC-FA improves some macro values but fails the registered HELD tail and wall-clock gates in the first 3090 round.
   - Source: `feedback_geometry_pipeline_2026-11-04/NARRATIVE_REPORT.md`, `PAPER_CLAIM_AUDIT_PIVOT_2026-11-05.md`, `PAPER_CLAIM_AUDIT_COST_AWARE_2026-11-05.md`.

4. **Systems turn**
   - Fused execution removes duplicate encoder work while preserving action sequences, logical counters and state values within the registered tolerance.
   - This is a systems implementation result that can be narrated alongside, but separately from, the algorithmic risk story.
   - Source: `diagnostics/gpu_tfgps/pcfa_fused_latency_2026-11-05/PCFA_FUSED_LATENCY_REPORT_2026-11-05.md`, `PCFA_FUSED_CLAIM_AUDIT_2026-11-06.md`.

5. **A100 action-space study**
   - A100 pilot expanded the registered action grid and executed the preflight, selection, freeze, HELD and deterministic audit chain.
   - The original action-space audit had a wall-clock gate failure at one configuration and did not support a universal algorithm claim.
   - Source: remote A100 output was mirrored under `PlantGPU/remote_full_mirror_20261107` where available, with the final deterministic audit in the GPU diagnostics tree.

6. **Partition heterogeneity gate**
   - FIT-only gate passed because partition macro-F1 gap was `0.0139206464`, above the registered `0.01` threshold.
   - Tail gap was `0.0029066` and action total variation was `0.0289855`, both below their thresholds.
   - The gate therefore triggered registration of PR-PV-MFAPS based on macro heterogeneity only.
   - Source: remote `out/bs_pcfa_plugin_2026-11-06/partition_heterogeneity_gate/PR_PARTITION_HETEROGENEITY_GATE.json`.

7. **PR-PV-MFAPS mathematical extension**
   - Shared policy across partitions.
   - KL ambiguity objective `Phi_epsilon(v) = sup_{u: KL(u||ubar)<=epsilon} u^T v` with log-sum-exp dual.
   - Conditional stability envelope from strongly convex/smooth inner gradient descent.
   - Finite-action Pareto DP and quantization residual accounting.
   - Source: `PR_PV_MFAPS_DERIVATION_PACKAGE_2026-11-07.md`, `MINIMAX_KAPPA_PROPOSITION.md`, remote/local PR diagnostics.

8. **Epsilon sweep and geometry-controlled action changes**
   - Eleven epsilon values produced five distinct policies.
   - At epsilon `0.01`, only two of 69 classes changed action relative to epsilon `0.1`.
   - The p0/p1 class counts are paired and identical, so the policy changes are explained by partition-specific risk envelopes rather than frequency differences.
   - Source: `diagnostics/gpu_tfgps/pr_pv_mfaps_2026-11-07/EPSILON_SWEEP_SUMMARY.json` and the local action-difference analysis.

9. **Final PR execution**
   - Clean PR-only run: six states across two partitions and three seeds.
   - Clean controls: 18 states for BS fixed-k and effective-number controls.
   - Final merged freeze: 24 artifacts and 24 unique states.
   - One-time HELD: 24 states, 25,104 reads, zero errors.
   - Main fixed-k gate: HELD Macro-F1 `+0.0287771`, HELD tail-10% `+0.0317460`, maximum wall ratio `0.997414`, maximum FLOPs ratio `0.888889`.
   - PR versus effective-number gains are reported separately because wall ratio is `1.076813`, above the fixed budget threshold.
   - `bs_pcfa` in the PR matrix reused the same eps=0.01 policy and is a duplicate sanity control, not an independent baseline.
   - Source: `diagnostics/gpu_tfgps/pr_pv_mfaps_2026-11-07/PR_PV_MFAPS_CLAIMS_EVIDENCE_AUDIT.json`, `.md`, `PR_FIT_FREEZE_MANIFEST.json`, `HELD_ONE_TIME_EVAL_VALID_EPS001.json`.

## Executed evidence

- Original PlantSPD E0/E1/B3/E2 FIT-only endpoint families.
- TFGPS deterministic contracts and local audit.
- 3090 pivot and cost-aware PC-FA rounds with freeze and one-time HELD.
- A100 preflight, action-space pilot, fused equivalence and final chain.
- PR mathematical smoke test and KL objective checks.
- Epsilon sweep and FIT-only policy selection at epsilon `0.01`.
- Clean PR matched run, clean controls, 24-state freeze, one-time HELD and claims audit.

## Registered but not executed

- CIFAR-LT, CIFAR100-LT and ImageNet-LT transfer.
- Learnable spectral geometry / `Power(p)` comparison.
- Independent second SPD project feedback-transfer run.
- Independent semantic reviewer.
- Any additional second-domain matched PR run.

## Unverified claims

- Community or absolute SOTA.
- Universal long-tail improvement.
- Cross-task generalization.
- Alpha-aware optimality of the discrete policy.
- A robust-cost theorem for the shared-cost PR variant.
- A neural-network-wide population generalization theorem.

## Quarantined or invalid attempts

- Early PC-FA and K-MFAPS artifacts that violated the registered configuration.
- First PR smoke with wrong Python/environment path.
- Epsilon `0.1` smoke stopped after p0 because it was a policy platform and was not used for final selection.
- First merged PR matched run with PR/control state-name collision, yielding only 18 independent artifacts.
- Second v2 run with the same internal state collision.
- First PR HELD attempt with invalid state paths.
- These artifacts should be shown as plot twists in the narrative, but never aggregated into final metrics.

## Visualization candidates

- Timeline from Raw/Log mechanism discovery to long-tail scheduling and PR-DRO.
- Macro-F1 versus tail-10% Pareto scatter for native, effective-number, fixed-k, K-MFAPS, PC-FA and PR.
- Epsilon versus number of changed actions.
- Partition risk envelopes for the two changed classes.
- Legacy versus fused physical encoder forward count and wall/FLOPs ratios.
- State-machine diagram for FIT calibration, freeze, HELD and claims classification.
- Action histograms for each allocation family.
