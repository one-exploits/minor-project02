import mne

edf_file = "SC4001E0-PSG.edf"
#edf_file="SC4001EC-Hypnogram.edf"  
raw = mne.io.read_raw_edf(edf_file, preload=True)
img=raw.plot(n_channels=7, scalings='auto', title="EEG Data", block=True)
img.savefig("signals_plot.png")

print(raw.info)

print("hellwo")


channel_n = raw.ch_names
print("Channels:", channel_n)
data, times = raw[:]

fp1_index = channel_n.index('EEG Pz-Oz')
fp1_signal = data[fp1_index, :] 

def plot_specific_channel(file,channels_name):
    raw = mne.io.read_raw_edf(file, preload=True)
    slected_channel=raw.copy().pick_channels(channels_name);
    slected_channel.plot(n_channels=len(channels_name),scalings='auto', title='EEG, EOG and EMG signals',block=True);
    return ;



specific_channels = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]
plot_specific_channel(edf_file,specific_channels);
# import matplotlib.pyplot as plt
# plt.plot(times, fp1_signal)
# plt.xlabel("Time (s)")
# plt.ylabel("EEG Signal (µV)")
# plt.title("Fp1 Channel EEG Signal")
# plt.show()
