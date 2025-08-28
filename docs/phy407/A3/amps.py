import numpy as np
def cosine_wave(time, amplitude, period, phase):
    """Generate a sine wave with known amplitude, period, and phase"""
    return amplitude * np.cos((time/period + phase) * 2 * np.pi)

t = np.arange(0, 100, .1)
y = cosine_wave(t, 2, 20, 0.)
fft1 = np.fft.fft(y)
fft2 = np.fft.ifft(y)
fft3 = np.fft.rfft(y)
fft4 = np.fft.irfft(y)
amp1 = np.abs(fft1)
amp2 = np.abs(fft2)
amp3 = np.abs(fft3)
amp4 = np.abs(fft4)

# Are any of the following numbers what you expect?
print(amp1.max(), amp2.max())
print(amp3.max(), amp4.max())