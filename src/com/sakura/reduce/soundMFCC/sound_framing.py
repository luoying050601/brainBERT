import numpy as np
import scipy.io.wavfile
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000

# 原始数据,读取前3.5s 的数据
sample_rate, signal = scipy.io.wavfile.read('wonderland_ch_01.wav')
original_signal = signal[0:int(3.5 * sample_rate)]

signal_num = np.arange(len(signal))
sample_num = np.arange(len(original_signal))

# 预加重
pre_emphasis = 0.97
emphasized_signal = np.append(original_signal[0], original_signal[1:] - pre_emphasis * original_signal[:-1])
emphasized_signal_num = np.arange(len(emphasized_signal))

# 分帧
frame_size = 0.025
frame_stride = 0.1
frame_length = int(round(frame_size * sample_rate))
frame_step = int(round(frame_stride * sample_rate))
signal_length = len(emphasized_signal)
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

pad_signal_length = num_frames * frame_step + frame_length
pad_signal = np.append(emphasized_signal, np.zeros((pad_signal_length - signal_length)))

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
    np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]

# 绘图 03
plt.figure(figsize=(11, 4), dpi=500)

pad_signal_num = np.arange(len(pad_signal))
plt.plot(pad_signal_num / sample_rate, pad_signal, color='blue')
plt.xlabel("Time (sec)", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.title("pad signal of Voice", fontsize=14)

plt.savefig('mfcc_03.png')
