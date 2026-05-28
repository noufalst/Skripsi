import new_baru_revised_same_structure_v2 as gr
import matplotlib.pyplot as plt

ni_all = gr.load_multiple_NI_sensor_data(base_dir=".")
w = ni_all.loc["2026-03-31 11:58":"2026-04-02 21:42"]

cols = ["T1Ta", "T1Tb", "T1Ka", "T1Ke", "T2Ka"]
w[cols].plot(figsize=(14, 6), grid=True)
plt.title("CAM and RR NI Sensor Channels Check")
plt.ylabel("Temperature (°C)")
plt.show()