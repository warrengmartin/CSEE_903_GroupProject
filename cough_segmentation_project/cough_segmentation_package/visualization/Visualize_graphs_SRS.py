import numpy as np
import matplotlib.pyplot as plt
import librosa

def compute_mel_spectrogram(x, fs, n_mels=128, n_fft=2048, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(y=x, sr=fs, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Define hop_length in the global scope
hop_length = 512

# Load the two cough audio files - paste in audio files
file1 = "/kaggle/input/audio-file-dataset-roneel/0969d0c4-34ce-4e9a-8cf1-1b18403587e8.wav"
file2 = "/kaggle/input/audio-file-dataset-roneel/09e64861-61b1-4f66-a7c0-17ac1a0c60f0.wav"

# Load the audio signals
x1, fs1 = librosa.load(file1, sr=None)
x2, fs2 = librosa.load(file2, sr=None)

# Compute Mel spectrograms
mel_spec_db1 = compute_mel_spectrogram(x1, fs1, hop_length=hop_length)
mel_spec_db2 = compute_mel_spectrogram(x2, fs2, hop_length=hop_length)

# Create a figure with two rows and two columns
fig, axs = plt.subplots(2, 2, figsize=(12, 6))

# Calculate time arrays for the waveforms
time1 = np.arange(len(x1)) / fs1
time2 = np.arange(len(x2)) / fs2

# Plot the waveforms with time in seconds
axs[0, 0].plot(time1, x1)
axs[0, 0].set_title("(a) Cough Signal One")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Amplitude")
axs[0, 1].plot(time2, x2)
axs[0, 1].set_title("(b) Cough Signal Two")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Amplitude")

# Calculate time arrays for the spectrograms
time_spec1 = np.arange(mel_spec_db1.shape[1]) / (fs1 / hop_length)
time_spec2 = np.arange(mel_spec_db2.shape[1]) / (fs2 / hop_length)

# Plot the spectrograms with time in seconds
axs[1, 0].imshow(mel_spec_db1, origin='lower', aspect='auto', cmap='jet', interpolation='nearest', extent=[time_spec1[0], time_spec1[-1], 0, 128])
axs[1, 0].set_title("(c) Spectrogram of Sample One")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Mel Frequency")
axs[1, 1].imshow(mel_spec_db2, origin='lower', aspect='auto', cmap='jet', interpolation='nearest', extent=[time_spec2[0], time_spec2[-1], 0, 128])
axs[1, 1].set_title("(d) Spectrogram of Sample Two")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Mel Frequency")

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.4, hspace=0.6)

# Show the plot
plt.show()
