# -*- coding: utf-8 -*-
# @Time    : 11/12/20 3:17 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import matplotlib.pyplot as plt
import numpy as np
import random
from jjnutils.util import load_itk

img, _, _ = load_itk("/data/jjia/mt/data/lesion/valid/ori_ct/Covid_lesion/volume-covid19-A-0698.nii.gz")
img = img[20, :, :, ...]
noise = np.random.normal(0, 0.05, img.shape)

img_addnnoise = img + noise

plt.figure()
plt.subplot(3,3,1)
plt.imshow(noise)
plt.subplot(3,3,2)
plt.imshow(img)
plt.subplot(3,3,3)
plt.imshow(img_addnnoise)

normized_img  = (img - np.mean(img)) / np.std(img)
normized_img_addnoise = normized_img + noise
plt.subplot(3,3,4)
plt.imshow(noise)
plt.subplot(3,3,5)
plt.imshow(normized_img)
plt.subplot(3,3,6)
plt.imshow(normized_img_addnoise)

noise_2 = np.random.normal(0, 0.1, img.shape)
normized_img_addnoise2 = normized_img + noise_2
plt.subplot(3,3,7)
plt.imshow(noise_2)
plt.subplot(3,3,8)
plt.imshow(normized_img)
plt.subplot(3,3,9)
plt.imshow(normized_img_addnoise2)

plt.show()


print('a')