# CAM Plot Talking Points for Professor

## Main message

The model should not be judged only from `T1Tb vs T_s_in`, because measured `T1Tb` is much more dynamic than the 1D slab-bottom model node.

For tomorrow, present `T2A vs T_s_in` as the temporary damped-path validation target, with the caveat that physical sensor mapping still needs plug-off verification.

## Plot 1 — Main validation: T2A vs T_s_in

Raw:
- RMSE = 2.88 °C
- Bias = 2.83 °C
- Corr = 0.57
- Amplitude error = -0.25 °C

Bias-corrected:
- RMSE = 0.57 °C
- Bias ≈ 0.00 °C
- Corr = 0.57
- Amplitude error = -0.25 °C

Interpretation: the model captures a damped thermal response reasonably after removing systematic offset. This is not hidden calibration; the bias correction is reported separately.

## Plot 2 — Diagnostic: T1Tb vs T_s_in

- Measured T1Tb amplitude = 10.08 °C
- Model T_s_in amplitude = 1.99 °C
- Amplitude error = -8.09 °C

Interpretation: T1Tb is too dynamic to be represented by the current 1D slab-bottom node. This suggests sensor/path mismatch, edge conduction, or a heat path not captured by the model.

## Suggested wording

“Pak, saya awalnya validasi CAM terhadap sensor atap indoor T1Tb, tetapi hasil model terlalu flat. Saya kemudian lakukan stack-aware validation. Hasilnya menunjukkan T1Tb kemungkinan tidak merepresentasikan node T_s_in model 1D secara bersih karena amplitudonya jauh lebih besar. Untuk sementara saya gunakan T2A sebagai target validasi damped thermal response karena bentuk dan amplitudonya lebih dekat dengan node model. Namun mapping sensor masih perlu saya verifikasi dengan plug-off test.”

## Next step

- Plug-off test sensor untuk memastikan channel fisik.
- Cek apakah T1Ta/T1Tb terkena side/edge heat path.
- Setelah mapping fix, rerun stack-aware validation.
