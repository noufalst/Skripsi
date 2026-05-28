# CAM + C3 talking points

## CAM
Use `01_CAM_main_T2A_vs_Ts_in.png`.

CAM bias-corrected:
- RMSE = 0.57 °C
- Corr = 0.57
- Amplitude error = -0.25 °C

## C3
Use `02_C3_validation.png`.

C3 bias-corrected:
- RMSE = 1.42 °C
- Corr = 0.10
- Amplitude error = 0.70 °C

## Suggested wording
Pak, untuk CAM saya gunakan T2A sebagai target sementara validasi damped thermal response karena T1Tb terlalu dinamis terhadap node model T_s_in. Untuk C3, saya jalankan validasi terpisah dengan target yang paling cocok terhadap node model, sambil tetap menyatakan bahwa mapping sensor perlu diverifikasi fisik.
