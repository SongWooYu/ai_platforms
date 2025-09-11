import os
from PIL import Image
from matplotlib import pyplot as plt

# path_dir = './images/training/'
# file_list = os.listdir(path_dir)
# file_list.sort()
# print(file_list)

# for i in range(0, 10):
#     path_dir = './images/training/{0}'.format(i)
#     file_list = os.listdir(path_dir)
#     file_list.sort()
#     print(file_list)

all_files = []
for i in range(0, 10):
    path_dir = './images/training/{0}'.format(i)
    file_list = os.listdir(path_dir)
    file_list.sort()
    all_files.append(file_list)
num = 2
target = 50
img = Image.open('./images/training/{0}/'.format(num) + all_files[num][target])
# img.show()

plt.imshow(img)
plt.show()

