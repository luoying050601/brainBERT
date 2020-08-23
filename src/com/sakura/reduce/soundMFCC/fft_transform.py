import numpy as np
import scipy.io.wavfile
from matplotlib import pyplot as plt
import matplotlib as mpl
import cv2

mpl.rcParams['agg.path.chunksize'] = 10000

# 原始数据,读取5-6.5s 的数据
sample_rate, signal = scipy.io.wavfile.read('wonderland_ch_01.wav')
original_signal = signal[5:int(1.5 * sample_rate)]

signal_num = np.arange(len(signal))
sample_num = np.arange(len(original_signal))

# 汉明窗
N = 200
x = np.arange(N)
y = 0.54 * np.ones(N) - 0.46 * np.cos(2 * np.pi * x / (N - 1))

# plt.plot(x, y, label='Hamming')
# plt.xlabel("Samples")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.savefig('hamming.png', dpi=500)
# 预加重
pre_emphasis = 0.97
emphasized_signal = np.append(original_signal[0], original_signal[1:] - pre_emphasis * original_signal[:-1])
emphasized_signal_num = np.arange(len(emphasized_signal))

# 分帧
frame_size = 0.025
frame_stride = 0.1
frame_length = int(round(frame_size * sample_rate))
frame_step = int(round(frame_stride * sample_rate))
print(frame_step)
signal_length = len(emphasized_signal)
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

pad_signal_length = num_frames * frame_step + frame_length
pad_signal = np.append(emphasized_signal, np.zeros((pad_signal_length - signal_length)))

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
    np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]

# 加汉明窗
frames *= np.hamming(frame_length)
# Explicit Implementation
# frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))

# 傅里叶变换和功率谱
NFFT = 512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = (1.0 / NFFT) * (mag_frames ** 2)

# 将频率转换为Mel频率
low_freq_mel = 0

nfilt = 40
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz

bin = np.floor((NFFT + 1) * hz_points / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))

for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])  # left
    f_m = int(bin[m])  # center
    f_m_plus = int(bin[m + 1])  # right
    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
#幅度转化成分贝（dB）
filter_banks = 20 * np.log10(filter_banks)  #

num_ceps = 12
mfcc = cv2.dct(filter_banks)[:, 1: (num_ceps + 1)]
(nframes, ncoeff) = mfcc.shape

n = np.arange(ncoeff)
cep_lifter = 22
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= lift

# 为了平衡频谱并改善信噪比（SNR），我们可以简单地从所有帧中减去每个系数的平均值。平均归一化滤波器组
filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
# 平均归一化MFCC
mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

# 绘图 04
plt.figure(figsize=(11, 7), dpi=500)

plt.subplot(211)
plt.imshow(np.flipud(filter_banks.T), cmap=plt.cm.jet, aspect=0.2,
           extent=[0, filter_banks.shape[1], 0, filter_banks.shape[0]])  # 画热力图
plt.title("MFCC")

plt.subplot(212)
plt.imshow(np.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.2, extent=[0, mfcc.shape[0], 0, mfcc.shape[1]])  # 热力图
plt.title("MFCC")

plt.savefig('mfcc_04.png')
