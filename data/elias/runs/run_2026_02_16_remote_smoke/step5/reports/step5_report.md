# Step 5 Report (run_2026_02_16_remote_smoke)

## Recovery-Aware Conclusion
Recovery-aware conclusion: level=weak; step3_soft_gate=weak; step4_group_winner=cont_asymptote; step4_vote_tie=False; ppc_mean_joint_nll_per_trial=17.863966; hazard_mean_h_shrinkage_spearman=-0.142538; latent_mean_choice_accuracy=0.551887; latent_mean_timeout_rate=0.065697.

## Cross-Step Context
- Step 3 soft-gate status: `weak`
- Step 4 group winner: `cont_asymptote`
- Step 4 vote tie: `False`

## Step 5 Summary Metrics
- PPC mean joint NLL per trial: `17.86396646832399`
- Hazard-signature mean Spearman(H, shrinkage): `-0.1425382301118157`
- Latent mean choice accuracy (excluding timeout): `0.5518865810759338`
- Latent mean timeout rate: `0.06569739653621234`

## Step 5 Artifact Counts
- Posterior predictive blocks: `12`
- Hazard-signature blocks: `12`
- Latent-summary blocks: `12`

## Hazard Input Caveat
Caveat: Hazard input was fixed to `subjective_h_snapshot` and treated as an externally provided past-only signal during fit and evaluation; this does not establish endogenous hazard inference by the models.
