# Verification report

Generated 2026-11-01 10:27 by `scripts/verify_all.py`.

**Summary: 75 PASS, 1 FAIL.**


## 1. Physics and bookkeeping tests (tests/test_physics.py)

| Result | Check | Detail |
|---|---|---|
| PASS | test_bfs_vs_pandapower | cases=50, max_v_err_pu=1.829e-10, max_loss_err_kw=1.759e-07, base_vmin_pu=0.9131, base_vmin_bus=18, base_loss_kw=202.7 |
| PASS | test_lfc_steady_state | df_sim_hz=-0.02381, df_theory_hz=-0.02381, tie_sim_mw=-5, tie_theory_mw=-5, agc_df_end_hz=-5.851e-09, agc_tie_end_mw=-1.147e-06 |
| PASS | test_regulation_energy_integral | e_out_mwh=0.01667, e_in_mwh=0.008333, batt_mwh=-0.01045, expected_batt_mwh=-0.01045 |
| PASS | test_feasibility_random_policy | steps=960, max_power_excess_mw=0, max_reserve_excess_mw=0, min_sustained_margin_mwh=0, max_energy_violation_mwh=0, soc_min=0.2, soc_max=0.8348, max_energy_closure_err_mwh=1.776e-15, raw_clip_components=1576, projection_events=37 |
| PASS | test_feasibility_fullhour | steps=144, max_power_excess_mw=0, max_reserve_excess_mw=0, min_sustained_margin_mwh=0, max_energy_violation_mwh=0, soc_min=0.2, soc_max=0.9, max_energy_closure_err_mwh=3.553e-15, raw_clip_components=221, projection_events=10 |
| PASS | test_no_reserve_from_mobility_shortfall | cases=600, max_energy_violation_mwh=0, cases_with_reserve=465 |
| PASS | test_clip_counters_distinct | case1_raw=[False, False, False, False], case1_proj=[True, True], case2_raw=[True, False, True, False] |

## 2. IEEE 33-bus verification against pandapower (scripts/validate_models.py re-run)

| Result | Check | Detail |
|---|---|---|
| PASS | validate_models.py runs | return code 0 |
| PASS | max |V_BFS - V_pandapower| < 1e-8 p.u. | 1.220e-10 p.u. over 20 random cases |
| PASS | max loss difference < 1e-5 kW | 5.241e-08 kW |
| PASS | base case Vmin = 0.9131 p.u. at bus 18 | 0.91309 p.u., bus 18 |
| PASS | base case loss = 202.68 kW | 202.677 kW |
| PASS | LFC steady-state df = -0.0238095 Hz | sim -0.0238095238 Hz, theory -0.0238095238 Hz |
| PASS | LFC steady-state dPtie = -5 MW | sim -5.000000000 MW |

## 3. Provenance: results come from the corrected model

| Result | Check | Detail |
|---|---|---|
| PASS | every evaluation record contains regulation-energy and clipping fields | 31 files |
| PASS | energy balance closes in every evaluated hour (max |error| < 1e-9 MWh) | max 3.6e-15 MWh over 1860 controller-days |
| PASS | scheduled plus droop power never exceeds charger or energy limits | max excess 0.000e+00 MW |
| PASS | droop power never exceeds the committed reserve |  |
| PASS | no SoC-limit violation caused by regulation |  |
| PASS | sustained-activation margin never negative | min margin -4.441e-16 MWh |
| PASS | raw-action clipping and projection counted separately |  |
| PASS | environment default reg_energy_mode = 'decomposed' and ENV_VERSION = v3 |  |
| PASS | all evaluation files newer than the corrected environment | none stale |
| PASS | ppo_main_s0 trained in environment v3 (meta.env_version) | env_version=v3, wall 27 min |
| PASS | ppo_main_s1 trained in environment v3 (meta.env_version) | env_version=v3, wall 27 min |
| PASS | ppo_main_s2 trained in environment v3 (meta.env_version) | env_version=v3, wall 36 min |
| PASS | ppo_main_s3 trained in environment v3 (meta.env_version) | env_version=v3, wall 36 min |
| PASS | ppo_main_s4 trained in environment v3 (meta.env_version) | env_version=v3, wall 27 min |
| PASS | ppo_noDeg_s0 trained in environment v3 (meta.env_version) | env_version=v3, wall 12 min |
| PASS | ppo_noDeg_s1 trained in environment v3 (meta.env_version) | env_version=v3, wall 12 min |
| PASS | ppo_noDeg_s2 trained in environment v3 (meta.env_version) | env_version=v3, wall 12 min |
| PASS | ppo_noF_s0 trained in environment v3 (meta.env_version) | env_version=v3, wall 13 min |
| PASS | ppo_noF_s1 trained in environment v3 (meta.env_version) | env_version=v3, wall 14 min |
| PASS | ppo_noF_s2 trained in environment v3 (meta.env_version) | env_version=v3, wall 12 min |
| PASS | ppo_noOV_s0 trained in environment v3 (meta.env_version) | env_version=v3, wall 29 min |
| PASS | ppo_noOV_s1 trained in environment v3 (meta.env_version) | env_version=v3, wall 22 min |
| PASS | ppo_noOV_s2 trained in environment v3 (meta.env_version) | env_version=v3, wall 14 min |
| PASS | ppo_ref200_s0 trained in environment v3 (meta.env_version) | env_version=v3, wall 25 min |
| PASS | ppo_ref200_s1 trained in environment v3 (meta.env_version) | env_version=v3, wall 29 min |
| PASS | ppo_ref200_s2 trained in environment v3 (meta.env_version) | env_version=v3, wall 23 min |
| PASS | ppo_squash_s0 trained in environment v3 (meta.env_version) | env_version=v3, wall 15 min |
| PASS | ppo_squash_s1 trained in environment v3 (meta.env_version) | env_version=v3, wall 14 min |
| PASS | ppo_squash_s2 trained in environment v3 (meta.env_version) | env_version=v3, wall 27 min |
| PASS | ppo_wF2_s0 trained in environment v3 (meta.env_version) | env_version=v3, wall 19 min |
| PASS | ppo_wF2_s1 trained in environment v3 (meta.env_version) | env_version=v3, wall 12 min |
| PASS | ppo_wF8_s0 trained in environment v3 (meta.env_version) | env_version=v3, wall 12 min |
| PASS | ppo_wF8_s1 trained in environment v3 (meta.env_version) | env_version=v3, wall 12 min |
| PASS | ppo_wV100_s0 trained in environment v3 (meta.env_version) | env_version=v3, wall 12 min |
| PASS | ppo_wV100_s1 trained in environment v3 (meta.env_version) | env_version=v3, wall 19 min |
| PASS | ppo_wV25_s0 trained in environment v3 (meta.env_version) | env_version=v3, wall 12 min |
| PASS | ppo_wV25_s1 trained in environment v3 (meta.env_version) | env_version=v3, wall 12 min |
| PASS | five final main seeds 0-4 present | seeds [0, 1, 2, 3, 4] |

## 4. Manuscript numbers recomputed independently from raw results

| Result | Check | Detail |
|---|---|---|
| PASS | regulation-energy treatment evaluations present for all five treatments | results/eval_modes/* |
| PASS | all recomputed macros equal the manuscript values | 571 macros recomputed; 0 mismatches |

Macros not independently recomputed (410; formatting helpers, counts of seeds/days, learning-curve statistics checked in section 5): AblMainCostDeg, AblMainCostF, AblMainCostRest, AblMainCostTerm, AblMainCostTot, AblMainCostV, AblNoDegConvMax, AblNoDegCostDeg, AblNoDegCostF, AblNoDegCostRest, AblNoDegCostSd, AblNoDegCostTerm, AblNoDegCostTot, AblNoDegCostV, AblNoDegNadirMeanHzSd, AblNoDegOvPuH, AblNoDegOvPuHSd, AblNoDegPairDiffs, AblNoDegPairMean, AblNoDegPval, AblNoDegReserveMeanMwSd, AblNoDegReserveRedPct, AblNoDegSeedCosts, AblNoDegSeedSocDefs, AblNoDegSocDefMaxSeed, AblNoDegSocDefPctSd, AblNoDegThroughputMwhSd, AblNoDegUvPuHSd, AblNoFConvMax, AblNoFCostDeg, AblNoFCostF, AblNoFCostRest, AblNoFCostSd, AblNoFCostTerm, AblNoFCostTot, AblNoFCostV, AblNoFNadirMeanHzSd, AblNoFOvPuH, AblNoFOvPuHSd, AblNoFPairDiffs, AblNoFPairMean, AblNoFPval, AblNoFReserveMeanMwSd, AblNoFReserveRedPct, AblNoFSeedCosts, AblNoFSeedSocDefs, AblNoFSocDefMaxSeed, AblNoFSocDefPctSd, AblNoFThroughputMwhSd, AblNoFUvPuHSd, AblNoOVConvMax, AblNoOVCostDeg, AblNoOVCostF, AblNoOVCostRest, AblNoOVCostSd, AblNoOVCostTerm, AblNoOVCostTot, AblNoOVCostV, AblNoOVNadirMeanHzSd, AblNoOVOvPuH, AblNoOVOvPuHSd, AblNoOVPairDiffs, AblNoOVPairMean, AblNoOVPval, AblNoOVReserveMeanMwSd, AblNoOVReserveRedPct, AblNoOVSeedCosts, AblNoOVSeedSocDefs, AblNoOVSocDefMaxSeed, AblNoOVSocDefPctSd, AblNoOVThroughputMwhSd, AblNoOVUvPuHSd, AblRefConvMax, AblRefCostDeg, AblRefCostF, AblRefCostRest, AblRefCostSd, AblRefCostTerm, AblRefCostTot, AblRefCostV, AblRefNadirMeanHzSd, AblRefOvPuH, AblRefOvPuHSd, AblRefReserveMeanMwSd, AblRefReserveRedPct, AblRefSeedCosts, AblRefSeedSocDefs, AblRefSocDefMaxSeed, AblRefSocDefPctSd, AblRefThroughputMwhSd, AblRefUvPuHSd, AblSquashConvMax, AblSquashCostDeg, AblSquashCostF, AblSquashCostRest, AblSquashCostSd, AblSquashCostTerm, AblSquashCostTot, AblSquashCostV, AblSquashNadirMeanHzSd, AblSquashOvPuH, AblSquashOvPuHSd, AblSquashPairDiffs, AblSquashPairMean, AblSquashPval, AblSquashReserveMeanMwSd, AblSquashReserveRedPct, AblSquashSeedCosts, AblSquashSeedSocDefs, AblSquashSocDefMaxSeed, AblSquashSocDefPctSd, AblSquashThroughputMwhSd, AblSquashUvPuHSd, AblWfHighConvMax, AblWfHighCostDeg, AblWfHighCostF, AblWfHighCostRest, AblWfHighCostSd, AblWfHighCostTerm, AblWfHighCostTot, AblWfHighCostV, AblWfHighNadirMeanHzSd, AblWfHighOvPuH, AblWfHighOvPuHSd, AblWfHighPairDiffs, AblWfHighPairMean, AblWfHighPval, AblWfHighReserveMeanMwSd, AblWfHighReserveRedPct, AblWfHighSeedCosts, AblWfHighSeedSocDefs, AblWfHighSocDefMaxSeed, AblWfHighSocDefPctSd, AblWfHighThroughputMwhSd, AblWfHighUvPuHSd, AblWfLowConvMax, AblWfLowCostDeg, AblWfLowCostF, AblWfLowCostRest, AblWfLowCostSd, AblWfLowCostTerm, AblWfLowCostTot, AblWfLowCostV, AblWfLowNadirMeanHzSd, AblWfLowOvPuH, AblWfLowOvPuHSd, AblWfLowPairDiffs, AblWfLowPairMean, AblWfLowPval, AblWfLowReserveMeanMwSd, AblWfLowReserveRedPct, AblWfLowSeedCosts, AblWfLowSeedSocDefs, AblWfLowSocDefMaxSeed, AblWfLowSocDefPctSd, AblWfLowThroughputMwhSd, AblWfLowUvPuHSd, AblWvHighConvMax, AblWvHighCostDeg, AblWvHighCostF, AblWvHighCostRest, AblWvHighCostSd, AblWvHighCostTerm, AblWvHighCostTot, AblWvHighCostV, AblWvHighNadirMeanHzSd, AblWvHighOvPuH, AblWvHighOvPuHSd, AblWvHighPairDiffs, AblWvHighPairMean, AblWvHighPval, AblWvHighReserveMeanMwSd, AblWvHighReserveRedPct, AblWvHighSeedCosts, AblWvHighSeedSocDefs, AblWvHighSocDefMaxSeed, AblWvHighSocDefPctSd, AblWvHighThroughputMwhSd, AblWvHighUvPuHSd, AblWvLowConvMax, AblWvLowCostDeg, AblWvLowCostF, AblWvLowCostRest, AblWvLowCostSd, AblWvLowCostTerm, AblWvLowCostTot, AblWvLowCostV, AblWvLowNadirMeanHzSd, AblWvLowOvPuH, AblWvLowOvPuHSd, AblWvLowPairDiffs, AblWvLowPairMean, AblWvLowPval, AblWvLowReserveMeanMwSd, AblWvLowReserveRedPct, AblWvLowSeedCosts, AblWvLowSeedSocDefs, AblWvLowSocDefMaxSeed, AblWvLowSocDefPctSd, AblWvLowThroughputMwhSd, AblWvLowUvPuHSd, BestValCost, ClipAbovePsetOne, ClipAbovePsetTwo, ClipAboveRhoOne, ClipAboveRhoTwo, ClipBelowPsetOne, ClipBelowPsetTwo, ClipBelowRhoOne, ClipBelowRhoTwo, ClipDimAll, ClipDimSteps, ClipMedianExcess, ClipSquashAll, ConvChangePct, CrossDAOHi, CrossDAOLo, DayPPOPeakOne, DayRBNightVmin, DayResEveMin, DayResHi, DayResLo, FinalValCost, FinalValCostSd, FreqHour, HrHtwo, HrPPONadir, HrRBNadir, HrUCNadir, ModeDecomposedDAOEViolMwh, ModeDecomposedDAOViolHours, ModeDecomposedPPOEViolMwh, ModeDecomposedPPOViolHours, ModeDecomposedRBEViolMwh, ModeDecomposedRBViolHours, ModeDecomposedUCEViolMwh, ModeDecomposedUCViolHours, ModeDeltaCostDecomposed, ModeDeltaCostFullhour, ModeDeltaCostHour, ModeDeltaCostNone, ModeDeltaCostSdDecomposed, ModeDeltaCostSdFullhour, ModeDeltaCostSdHour, ModeDeltaCostSdNone, ModeDeltaCostSdWindow, ModeDeltaCostWindow, ModeFullhourDAOEViolMwh, ModeFullhourDAOViolHours, ModeFullhourPPOEViolMwh, ModeFullhourPPOViolHours, ModeFullhourRBEViolMwh, ModeFullhourRBViolHours, ModeFullhourUCEViolMwh, ModeFullhourUCViolHours, ModeHourDAOEViolMwh, ModeHourDAOViolHours, ModeHourPPOEViolMwh, ModeHourPPOViolHours, ModeHourRBEViolMwh, ModeHourRBViolHours, ModeHourUCEViolMwh, ModeHourUCViolHours, ModeNoneDAOEViolMwh, ModeNoneDAOViolHours, ModeNonePPOEViolMwh, ModeNonePPOViolHours, ModeNoneRBEViolMwh, ModeNoneRBViolHours, ModeNoneUCEViolMwh, ModeNoneUCViolHours, ModeWindowDAOEViolMwh, ModeWindowDAOViolHours, ModeWindowPPOEViolMwh, ModeWindowPPOViolHours, ModeWindowRBEViolMwh, ModeWindowRBViolHours, ModeWindowUCEViolMwh, ModeWindowUCViolHours, NSeed, NStress, NTest, PPOSeedSd, RawClipComponents, RawClipFracPct, RawClipPPct, RawClipRhoPct, RegClosureMax, RegDAOCumMax, RegDAOCumMin, RegDAOGrossDay, RegDAOSocAreaMax, RegDAOSocPct, RegPPOCumMax, RegPPOCumMin, RegPPOGrossDay, RegPPOSocAreaMax, RegPPOSocPct, RegRBCumMax, RegRBCumMin, RegRBGrossDay, RegRBSocAreaMax, RegRBSocPct, RegUCCumMax, RegUCCumMin, RegUCGrossDay, RegUCSocAreaMax, RegUCSocPct, RhoEveOne, RhoMidOne, RhoMidTwo, RhoNightOne, RhoNightTwo, SDAOEViol, SDAOPwExcess, SDAOUnmetMax, SDAOUnmetMean, SPPOEViol, SPPOPwExcess, SPPOUnmetMax, SPPOUnmetMean, SRBEViol, SRBPwExcess, SRBUnmetMax, SRBUnmetMean, SUCEViol, SUCPwExcess, SUCUnmetMax, SUCUnmetMean, TDAOEViol, TDAOPwExcess, TDAOSocDefMedian, TDAOSocDefZeroPct, TDAOUnmetMax, TDAOUnmetMean, TNadirPDAO, TNadirPRB, TNadirPUC, TNadirWinsDAO, TNadirWinsRB, TNadirWinsUC, TPPOEViol, TPPOPwExcess, TPPOSocDefMedian, TPPOSocDefZeroPct, TPPOUnmetMax, TPPOUnmetMean, TRBEViol, TRBPwExcess, TRBUnmetMax, TRBUnmetMean, TUCEViol, TUCPwExcess, TUCUnmetMax, TUCUnmetMean, TrainMin, TrainSteps, TrkDAOBindHours, TrkDAOGrossDec, TrkDAOGrossDecRatio, TrkDAOGrossFull, TrkDAOGrossHour, TrkDAOGrossHourRatio, TrkDAOGrossWin, TrkDAOMarginMin, TrkDAONetDec, TrkDAONetFull, TrkDAONetHour, TrkDAONetWin, TrkDAOSatFrac, TrkDAOSignFlipPct, TrkPPOBindHours, TrkPPOGrossDec, TrkPPOGrossDecRatio, TrkPPOGrossFull, TrkPPOGrossHour, TrkPPOGrossHourRatio, TrkPPOGrossWin, TrkPPOMarginMin, TrkPPONetDec, TrkPPONetFull, TrkPPONetHour, TrkPPONetWin, TrkPPOSatFrac, TrkPPOSignFlipPct, TrkRBBindHours, TrkRBGrossDec, TrkRBGrossDecRatio, TrkRBGrossFull, TrkRBGrossHour, TrkRBGrossHourRatio, TrkRBGrossWin, TrkRBMarginMin, TrkRBNetDec, TrkRBNetFull, TrkRBNetHour, TrkRBNetWin, TrkRBSatFrac, TrkRBSignFlipPct, VOneDAOCost, VOnePPOCost, VOneRBCost, VOneUCCost, ValDAO, ValDfTh, ValMode, ValRB, ValTieTh, ValUC, ValVerr


## 5. Tables, learning curves and figures

| Result | Check | Detail |
|---|---|---|
| PASS | table_test.tex: every cell equals recomputation | 68 cells; 0 mismatches |
| PASS | table_stress.tex: every cell equals recomputation | 48 cells; 0 mismatches |
| PASS | table_ablation.tex in manuscript identical to generated file |  |
| PASS | table_clip.tex in manuscript identical to generated file |  |
| PASS | table_modes.tex in manuscript identical to generated file |  |
| PASS | table_regenergy.tex in manuscript identical to generated file |  |
| PASS | table_s_runs.tex in manuscript identical to generated file |  |
| PASS | table_sensitivity.tex in manuscript identical to generated file |  |
| PASS | table_stress.tex in manuscript identical to generated file |  |
| PASS | table_test.tex in manuscript identical to generated file |  |
| PASS | results_macros.tex identical to generated macros.tex |  |
| PASS | fig_box.pdf regenerated after final evaluation and copied |  |
| PASS | fig_day.pdf regenerated after final evaluation and copied |  |
| PASS | fig_feeder.pdf regenerated after final evaluation and copied |  |
| PASS | fig_freq.pdf regenerated after final evaluation and copied |  |
| PASS | fig_heatmap.pdf regenerated after final evaluation and copied |  |
| PASS | fig_learning.pdf regenerated after final evaluation and copied |  |
| PASS | fig_policy.pdf regenerated after final evaluation and copied |  |
| PASS | fig_profiles.pdf regenerated after final evaluation and copied |  |
| PASS | FinalValCost | 137.7 |

## 6. Manuscript text: literal numbers outside macros

| Result | Check | Detail |
|---|---|---|
| **FAIL** | no result-like literal numbers outside the whitelist of model parameters | non-whitelisted literals: 10.5281 |
| PASS | manuscript compiles without undefined control sequences or references |  |
