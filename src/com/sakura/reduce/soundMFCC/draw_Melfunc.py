import numpy as np
import matplotlib.pyplot as plt

x = np.arange(8001)
y = 2595 * np.log10(1 + x / 700)

plt.plot(x, y, color='blue', linewidth=3)

plt.xlabel("f", fontsize=17)
plt.ylabel("Mel(f)", fontsize=17)
plt.xlim(0, x[-1])
plt.ylim(0, y[-1])

plt.savefig('mel_f.png', dpi=500)
