import numpy as np
import scipy.io.wavfile
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000
from scipy.fftpack import dct

# 原始数据,读取前3.5s 的数据
sample_rate, signal = scipy.io.wavfile.read('wonderland_ch_01.wav')
original_signal = signal[0:int(3.5 * sample_rate)]

signal_num = np.arange(len(signal))
sample_num = np.arange(len(original_signal))

# 绘图 01
plt.figure(figsize=(11, 7), dpi=500)

plt.subplot(211)
plt.plot(signal_num / sample_rate, signal, color='black')
plt.plot(sample_num / sample_rate, original_signal, color='blue')
plt.ylabel("Amplitude")
plt.title("signal of Voice")

plt.subplot(212)
plt.plot(sample_num / sample_rate, original_signal, color='blue')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude")
plt.title("3.5s signal of Voice ")

plt.savefig('mfcc_01.png')
