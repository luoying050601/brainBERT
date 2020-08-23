import scipy.io as scio
import os
import h5py

import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../../"))

data_path = DATA_DIR + "/original_data/EGG/proc/S01.mat"

data = scio.loadmat(data_path)
proc = data['proc'][0, 0]

print(proc[4].shape)
# 2129*6 = 12774
#  the epoch definitions (2,129; 919 content words and 1,210 function words


# print(proc[9][0][0][0][0][0][0][0][0][0][0][0][0].shape) #

for i in proc:
    print(i)

# <class 'dict'>
# print()
# raw = (data['raw'])[0]
# for id1, va1 in enumerate(raw):
#     print('shape', va1.shape)
#     print('type', type(va1))

# print(data['raw'])
# matlab_batch = data.get('matlabbatch')  # 取出字典里的label
