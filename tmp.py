import os
import numpy as np

np.random.seed(0)

# data_dir = ''
# os.makedirs(os.path.join(data_dir, 'scaled_sfm_depths'), exist_ok=True)
#
# for folder in os.listdir(os.path.join(data_dir, 'sfm_depths')):
#     for filename in os.listdir(os.path.join(data_dir, 'sfm_depths', folder)):
#         gt_depth = np.load(os.path.join(data_dir, 'depths', folder, filename))
#         pred_depth = np.load(os.path.join(data_dir, 'sfm_depths', folder, filename))
#
#         pred_depth = np.median(gt_depth) * pred_depth / np.median(pred_depth)
#
#         os.makedirs(os.path.join(data_dir, 'scaled_sfm_depths', folder), exist_ok=True)
#         np.save(os.path.join(data_dir, 'scaled_sfm_depths', folder, filename), pred_depth)



root_dir = '/mnt/storage/Projects/Pytorch-UNet/data/imgs'

all_folders = [folder for folder in os.listdir(root_dir)]
np.random.shuffle(all_folders)

train_len = int(np.floor(len(all_folders) * 0.7))
train_folders = all_folders[:train_len]
test_folders = all_folders[train_len:]


scannet_train_depth_path = '/mnt/storage/Projects/Indoor-SfMLearner/splits/scannet_train_depth.txt'

with open(scannet_train_depth_path, 'w') as f:
    for folder in train_folders:
        filepaths = [os.path.join(folder, filename) for filename in os.listdir(os.path.join(root_dir, folder))]
        filepaths = sorted(filepaths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        for idx in range(4, len(filepaths) - 4, 1):
            f.write('{} {} {} {} {} {} {} {} {}\n'.format(filepaths[idx], filepaths[idx-4], filepaths[idx-3],
                                                        filepaths[idx-2], filepaths[idx-1], filepaths[idx + 1],
                                                        filepaths[idx + 2], filepaths[idx + 3], filepaths[idx + 4]))


scannet_test_depth_path = '/mnt/storage/Projects/Indoor-SfMLearner/splits/scannet_test_depth.txt'

with open(scannet_test_depth_path, 'w') as f:
    for folder in test_folders:
        filepaths = [os.path.join(folder, filename) for filename in os.listdir(os.path.join(root_dir, folder))]
        filepaths = sorted(filepaths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        for idx in range(4, len(filepaths) - 4, 1):
            f.write('{} {} {} {} {} {} {} {} {}\n'.format(filepaths[idx], filepaths[idx-4], filepaths[idx-3],
                                                        filepaths[idx-2], filepaths[idx-1], filepaths[idx + 1],
                                                        filepaths[idx + 2], filepaths[idx + 3], filepaths[idx + 4]))