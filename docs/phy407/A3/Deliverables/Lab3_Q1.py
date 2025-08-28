from scipy.io.wavfile import read, write
from matplotlib import pyplot as plt
import numpy as np
from numpy import fft
from numpy import empty

sample, data = read('GraviteaTime.wav')
channel_0 = data[:, 0]
channel_1 = data[:, 1]
nsamples = len(channel_0)
time = np.linspace(0, nsamples/sample, nsamples) # Time array

# Part a)

# Channel 1
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time, channel_0, color='teal')
plt.title("Channel 0 of Original Audio Signal")
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.grid(True)

# Channel 2
plt.subplot(2, 1, 2)
plt.plot(time, channel_1, color='orange')
plt.title("Channel 1 of Original Audio Signal")
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.grid(True)


# Part b)

# First, we Fourier Transform the original time signal to the frequency domain
dft_0 = fft.fft(channel_0) / nsamples
dft_1 = fft.fft(channel_1) / nsamples
freqs = fft.fftfreq(nsamples, 1/sample) # frequencies

# Next, we apply a filter
filtered_dft_0 = np.copy(dft_0)
filtered_dft_0[np.abs(freqs) > 880] = 0

filtered_dft_1 = np.copy(dft_1)
filtered_dft_1[np.abs(freqs) > 880] = 0

# Now we apply an inverse transform to go back to the time domain
channel_0_filtered = np.real(fft.ifft(filtered_dft_0 * nsamples))
channel_1_filtered = np.real(fft.ifft(filtered_dft_1 * nsamples))


# Channel 0 - All plots in one figure
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(freqs, np.abs(dft_0), color='teal')
plt.title("DFT of Channel 0 of Original Audio Signal")
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")

plt.subplot(2, 2, 2)
plt.plot(freqs, np.abs(filtered_dft_0), color='teal')
plt.title("Filtered DFT of Channel 0 of Original Audio Signal")
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")

plt.subplot(2, 2, 3)
plt.plot(time, channel_0, color='teal')
plt.title("Channel 0 of Original Audio Signal (Time Domain)")
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")

plt.subplot(2, 2, 4)
plt.plot(time, channel_0_filtered, color='teal')
plt.title("Filtered Channel 0 (Time Domain)")
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")

plt.tight_layout()
plt.show()


# Channel 1 - All plots in one figure
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(freqs, np.abs(dft_1), color='orange')
plt.title("DFT of Channel 1 of Original Audio Signal")
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")

plt.subplot(2, 2, 2)
plt.plot(freqs, np.abs(filtered_dft_1), color='orange')
plt.title("Filtered DFT of Channel 1 of Original Audio Signal")
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")

plt.subplot(2, 2, 3)
plt.plot(time, channel_1, color='orange')
plt.title("Channel 1 of Original Audio Signal (Time Domain)")
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")

plt.subplot(2, 2, 4)
plt.plot(time, channel_1_filtered, color='orange')
plt.title("Filtered Channel 1 (Time Domain)")
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")

plt.tight_layout()


# Part c)

# Define a range of 30ms
num_samples = int(30/1000 * sample)

plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.title("Filtered Channel 0 (Time Domain)")
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.plot(time[:num_samples], channel_0_filtered[:num_samples], color="teal")

plt.subplot(2, 1, 2)
plt.title("Filtered Channel 1 (Time Domain)")
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.plot(time[:num_samples], channel_1_filtered[:num_samples], color="orange")
plt.show()


# Here we create and populate the output channel
data_out = empty(data.shape, dtype = data.dtype)
data_out[:, 0] = channel_0_filtered
data_out[:, 1] = channel_1_filtered

# Finally, we write the output array to a new .wav file
write('GraviteaTime_filtered.wav', sample, data_out)
