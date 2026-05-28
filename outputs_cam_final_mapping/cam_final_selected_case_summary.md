# CAM final sensor-mapping summary
Model module: new_baru_revised_same_structure_v3_t2a_subs_candidate_foliage_guard

Sensor interpretation used in this runner:
- T1Ka = confirmed inner roof / T_s_in target
- T1Ke = upper-zone indoor air proxy / T_air_upper
- T2A2 = candidate outer roof interface above slab, below substrate
- T2A = uncertain candidate, not preferred for outer roof

Selected CAM case:
- case: T1Ke_upper_air_hin_8
- h_in_W_m2K: 8.0
- rmse_C: 0.8440860053341978
- mae_C: 0.6832923867216296
- bias_C: 0.6138774581697998
- corr: 0.9088311271348872
- obs_amp_C: 4.846575000000001
- sim_amp_C: 4.589061985419789
- amp_error_C: -0.25751301458021203
- selection_score: 0.995634730840582

Interpretation:
- Main validation target is T1Ka because it is the confirmed inner-roof / indoor-side roof surface temperature.
- T1Ke is used as a dynamic upper-zone indoor air boundary, not as room-average air temperature.
- T2A2 is used as secondary observational evidence for outer-to-inner roof lag. It is not an exposed outdoor surface; it represents the above-slab / below-substrate interface candidate.
- h_in is swept because the cabin had no fan or ventilation, making natural/stagnant indoor convection more defensible.

Observed outer-to-inner lag diagnostic:
- xcorr_T2A2_leads_T1Ka_min: 110.0
- xcorr_best_corr: 0.8788283504069087
- daily_median_T2A2_leads_T1Ka_by_peak_min: 31.0
- T2A2_amp_ratio_vs_T1Ka: 1.3046914195503203
- corr_T2A2_with_G_sol: 0.624395911673478
- corr_T2A2_with_T_a: 0.8774378928044185

T1Ke upper-zone air diagnostic:
- corr_T1Ka_T1Ke: 0.8487636651933093
- mean_T1Ke_minus_T1Ka_C: 0.7412866394911825
- T1Ke_over_T1Ka_amp_ratio: 1.8432400200141323

Suggested thesis phrasing:
The CAM model was primarily validated against T1Ka, which represents the confirmed indoor-side roof surface temperature. T1Ke was applied as an upper-zone indoor air temperature proxy for the inner convective boundary because the sensor was suspended inside a closed, unventilated cabin. T2A2 was evaluated as a candidate outer-roof interface temperature above the slab and below the substrate. The observed phase lead of T2A2 relative to T1Ka indicates thermal delay through the roof/slab assembly.