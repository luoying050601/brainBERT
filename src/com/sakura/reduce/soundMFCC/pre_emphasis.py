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

# 绘图 02
plt.figure(figsize=(11, 4), dpi=500)

plt.plot(emphasized_signal_num / sample_rate, emphasized_signal, color='blue')
plt.xlabel("Time (sec)", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.title("emphasized signal of Voice", fontsize=14)

plt.savefig('mfcc_02.png')
