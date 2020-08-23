# not in use
import tensorlayer as tl
import numpy as np
import os
import nibabel as nib
import threading
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
from tensorlayer.prepro import *
import skimage.measure

nib.Nifti1Header.quaternion_threshold = - np.finfo(np.float32).eps * 10  # 松弛一下限制
training_data_path = "../../../original_data/fMRI/"
preserving_ratio = 0.25  # filter out 2d images containing < 25% non-zeros

f_train = tl.files.load_file_list(path=training_data_path,
                                  regx='.*.gz',
                                  printable=False)  # 将test测试集合中的数据以list形式存下来
X_train = []  # 处理训练集数据
for fi, f in enumerate(f_train):  # 相当于取出下标索引以及list里面相关的数据
    img_path = os.path.join(training_data_path, f)
    img = nib.load(img_path).get_fdata()
    print(img.shape)
    img_4d_max = np.amax(img)  # maximum of img
    print(img_4d_max)
    img = img / img_4d_max * 255  # 对所求的像素进行归一化变成0-255范围,这里就是4维数据
    print(img.shape[3])
    for i in range(img.shape[3]):  # 对切片进行循环
        img_3d = img[:, :, :, i]  # 取出一张图像
        # plt.imshow(img_3d)  # 显示图像
        # plt.pause(0.001)
        # # filter out 2d images containing < 10% non-zeros
        print(np.count_nonzero(img_3d))
       # print("before process:", img_3d.shape)
    #  if float(np.count_nonzero(img_2d)) / img_2d.size >= preserving_ratio:  # 表示一副图像非0个数超过整副图像的10%我们才把该图像保留下来
    #       img_2d = img_2d / 127.5 - 1  # 对最初的0-255图像进行归一化到[-1, 1]范围之内
    #       img_2d = np.transpose(img_2d, (1, 0))  # 这个相当于将图像进行旋转90度
    #       # plt.imshow(img_2d)
    #       # plt.pause(0.01)
    #       X_train.append(img_2d)
    # print(len(X_train))
# X_train = np.asarray(X_train, dtype=np.float32)  # 将训练的图像数据原来是list现在变成np.array格式
# X_train = X_train[:, :, :, np.newaxis]  # 变成4维数据
