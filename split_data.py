import os
import random
import glob
import shutil

test_percent = 0.1
val_percent = 0.1
# train_percent = 0.7
files = glob.glob('dataset/nuaa_64/*/*')

total_num = len(files)
test_num = int(total_num * test_percent)
val_num = int(total_num * val_percent)
# train_num = int(total_num * train_percent)

total_list = range(total_num)
test_val_list = random.sample(total_list, test_num + val_num)
test_list = random.sample(test_val_list, test_num)

for i in total_list:
    path = files[i]
    if i in test_val_list:
        if i in test_list:
            shutil.copy(path, os.path.join('dataset/test',path.split('/')[-2],path.split('/')[-1]))
        else:
            shutil.copy(path, os.path.join('dataset/val',path.split('/')[-2],path.split('/')[-1]))
    else:
        shutil.copy(path, os.path.join('dataset/train',path.split('/')[-2],path.split('/')[-1]))

