import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

all_files = []
for i in range(0, 10):
    path_dir = './images/training/{0}'.format(i)
    file_list = os.listdir(path_dir)
    file_list.sort()
    all_files.append(file_list)
num = 2
target = 50
img = Image.open('./images/training/{0}/'.format(num) + all_files[num][target])
img_arr = np.array(img)

# img.show()
# plt.imshow(img)
# plt.show()

print(img_arr)
print(img_arr.shape)