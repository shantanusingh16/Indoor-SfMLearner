# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import cv2
import random
import numpy as np
import copy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms

from datasets.extract_svo_point import PixelSelector


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('RGB'))


# class HabitatTestPoseDataset(data.Dataset):
#     def __init__(self,
#                  data_path,
#                  filenames,
#                  height,
#                  width,
#                  frame_idxs,
#                  is_train=False,
#                  ):
#         super(HabitatTestPoseDataset, self).__init__()
#         self.full_res_shape = (640, 480)
#         self.K = self._get_intrinsics()
#
#         self.data_path = data_path
#         self.filenames = filenames
#         self.height = height
#         self.width = width
#         self.interp = Image.ANTIALIAS
#
#         self.frame_idxs = frame_idxs
#         self.loader = pil_loader
#         self.to_tensor = transforms.ToTensor()
#
#         self.resize = transforms.Resize(
#                 (self.height, self.width),
#                 interpolation=self.interp
#         )
#
#         self.load_depth = False
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, index):
#         inputs = {}
#         line = self.filenames[index].split()
#         line = [os.path.join(self.data_path, item) for item in line]
#
#         for ind, i in enumerate(self.frame_idxs):
#             inputs[("color", i, -1)] = self.get_color(line[ind])
#
#         K = self.K.copy()
#         this_width = self.width
#         this_height= self.height
#
#         K[0, :] *= this_width
#         K[1, :] *= this_height
#
#         inv_K = np.linalg.pinv(K)
#
#         inputs[("K")] = torch.from_numpy(K).float()
#         inputs[("inv_K")] = torch.from_numpy(inv_K).float()
#
#         # self.preprocess(inputs)
#
#         for i in self.frame_idxs:
#             inputs[('color', i, 0)] = self.to_tensor(
#                     self.resize(inputs[('color', i,  -1)])
#             )
#             del inputs[("color", i, -1)]
#
#         if self.load_depth:
#             for ind, i in enumerate(self.frame_idxs):
#                 this_depth = line[ind].replace('color', 'depth').replace('.jpg', '.png')
#                 this_depth = cv2.imread(this_depth, -1) / 1000.0
#                 this_depth = cv2.resize(this_depth, (self.width, self.height))
#                 this_depth = self.to_tensor(this_depth)
#
#                 # assume no flippling
#                 inputs[("depth", i)] = this_depth
#
#         pose1_dir = line[0].replace('color', 'pose').replace('.jpg', '.txt')
#         pose2_dir = line[1].replace('color', 'pose').replace('.jpg', '.txt')
#         pose1 = np.loadtxt(pose1_dir, delimiter=' ')
#         pose2 = np.loadtxt(pose2_dir, delimiter=' ')
#         pose_gt = np.dot(np.linalg.inv(pose2), pose1)
#         inputs['pose_gt'] = pose_gt
#
#         return inputs
#
#     def get_color(self, fp):
#         color = self.loader(fp)
#         return Image.fromarray(color)
#
#     def check_depth(self):
#         return False
#
#     def _get_intrinsics(self):
#         w, h = self.full_res_shape
#         intrinsics =np.array([[640/w, 0., 320/w, 0.],
#                                [0., 480/h, 240/h, 0.],
#                                [0., 0., 1., 0.],
#                                [0., 0., 0., 1.]], dtype="float32")
#         return intrinsics



# class HabitatTestDepthDataset(data.Dataset):
#     def __init__(self,
#                  data_path,
#                  filenames,
#                  height,
#                  width,
#         ):
#         super(HabitatTestDepthDataset, self).__init__()
#
#         self.data_path = data_path
#         self.filenames = filenames
#         self.height = height
#         self.width = width
#         self.interp = Image.ANTIALIAS
#
#         self.loader = pil_loader
#         self.to_tensor = transforms.ToTensor()
#
#         self.resize = transforms.Resize(
#             (self.height, self.width),
#             interpolation=self.interp
#
#         )
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, index):
#         rgb = self.filenames[index].replace('/','_')
#         rgb = os.path.join(self.data_path, rgb)
#         depth = rgb.replace('color', 'depth').replace('jpg','png')
#
#         rgb = self.loader(rgb)
#         depth = cv2.imread(depth, -1) / 1000
#
#         rgb = Image.fromarray(rgb)
#
#         rgb = self.to_tensor(self.resize(rgb))
#         depth = self.to_tensor(depth)
#
#         return rgb, depth


class HabitatTrainDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        """

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 segment_path='',
                 return_segment=False,
                 shared_dict=None,
                 use_stereo=False):
        super(HabitatTrainDataset, self).__init__()

        self.debug = False
        self.use_stereo = use_stereo
        self.return_segment = return_segment
        # remove 16 pixels from borders
        self.full_res_shape = (640, 480)
        self.K = self._get_intrinsics()
        # print('The Normalized Intrinsics is ', self.K)

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.segment_path = segment_path
        self.img_cache = shared_dict

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()
        self.pixelselector = PixelSelector()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                # import pdb; pdb.set_trace()
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        if not (self.is_train):
            line = self.filenames[index].split()
            rgb_path = os.path.join(self.data_path, line[0])
            depth_path = os.path.join(self.data_path, line[1])
            # rgb, depth, norm, valid_mask = self.loader(line)

            # rgb = Image.fromarray(rgb)
            # depth = Image.fromarray(depth)

            rgb = self.get_color(rgb_path, False)
            depth = self.get_depth(depth_path, False)

            rgb = self.to_tensor(self.resize[0](rgb))
            depth = self.to_tensor(cv2.resize(depth, (self.width, self.height)))

            K = self.K.copy()
            K[0, :] *= self.width
            K[1, :] *= self.height
            return line[0], rgb, depth, K, np.linalg.pinv(K)

        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5

        # Disabled to avoid issues with stereo
        do_flip = False #self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        # line = [l.replace("/group/nyu_depth_v2", self.data_path) for l in line]
        line = [os.path.join(self.data_path, l) for l in line]
        for ind, i in enumerate(self.frame_idxs):
            if not i in set([0, -2, -1, 1, 2]):
                continue

            inputs[("color", i, -1)] = self.get_color(line[ind], do_flip)
            #inputs[("pose", i)] = torch.from_numpy(self.get_pose(line[ind], do_flip))
            if self.debug:
                inputs[("color", i, -1)] = self.to_tensor(self.get_color(line[ind], do_flip))

        if self.use_stereo:
            inputs[("color", "s", -1)] = self.get_stereo_right(line[0], do_flip)
            inputs[("cam_T_cam", 0, "s")] = torch.from_numpy(self.get_stereo_pose()).float()

        # load segments
        if self.return_segment:
            filename_wo_extn = os.path.splitext(os.path.basename(line[0]))[0]
            parent_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(line[0]))))
            sp = os.path.join(self.segment_path, parent_dir, 'seg_{}.npz'.format(filename_wo_extn))
            if self.img_cache is not None and sp in self.img_cache:
                segment = self.img_cache[sp]['segment_0']
            else:
                segment = cv2.resize(np.load(sp)['segment_0'], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                if self.img_cache is not None:
                    self.img_cache[sp] = {'segment_0': segment}

            # for ind, i in enumerate(self.frame_idxs):
            #     if not i in set([0]):
            #         continue
            #
            #     segment = segments['segment_%d' % (ind)]
            #     if do_flip:
            #         segment = cv2.flip(segment, 1)
            #     inputs[('segment', i, 0)] = self.to_tensor(segment).long() + 1

            if do_flip:
                segment = cv2.flip(segment, 1)
            inputs[('segment', 0, 0)] = self.to_tensor(segment).long() + 1

        if self.debug:
            return inputs

        # inputs[("color", 0, -1)] = self.get_color(line[0], do_flip)

        # svo_map = np.zeros((480, 640))
        svo_map_resized = np.zeros((self.height, self.width))  # 288 * 384
        img = np.array(inputs[("color", 0, -1)])
        key_points = self.pixelselector.extract_points(img)
        key_points = key_points.astype(int)
        key_points[:, 0] = key_points[:, 0] * self.height // 480
        key_points[:, 1] = key_points[:, 1] * self.width // 640

        # noise 1000 points
        noise_num = 3000 - key_points.shape[0]
        noise_points = np.zeros((noise_num, 2), dtype=np.int32)
        noise_points[:, 0] = np.random.randint(self.height, size=noise_num)
        noise_points[:, 1] = np.random.randint(self.width, size=noise_num)

        svo_map_resized[key_points[:, 0], key_points[:, 1]] = 1

        inputs['svo_map'] = torch.from_numpy(svo_map_resized.copy())

        svo_map_resized[noise_points[:, 0], noise_points[:, 1]] = 1
        inputs['svo_map_noise'] = torch.from_numpy(
            svo_map_resized,
        ).float()

        keypoints = np.concatenate((key_points, noise_points), axis=0)
        inputs['dso_points'] = torch.from_numpy(
            keypoints,
        ).float()

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            if not i in set([0, -2, -1, 1, 2, 's']):
                continue

            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        for key, val in inputs.items():
            if torch.any(torch.isnan(val) | torch.isinf(val)).cpu().item():
                print(self.filenames[index], key)

        return inputs

    def get_color(self, fp, do_flip):
        if self.img_cache is not None and fp in self.img_cache:
            color = self.img_cache[fp]
        else:
            color = self.loader(fp)
            if self.img_cache is not None:
                self.img_cache[fp] = color

        if do_flip:
            color = cv2.flip(color, 1)
            # color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return Image.fromarray(color)

    def get_stereo_right(self, left_fp, do_flip):
        right_fp = left_fp.replace('left_rgb', 'right_rgb')
        return self.get_color(right_fp, do_flip)

    def get_stereo_pose(self):
        T = np.eye(4)
        T[0, 3] = 0.2  # Baseline along x-axis with length 0.2m
        return T

    def get_pose(self, fp, do_flip):
        filename_wo_extn = os.path.splitext(os.path.basename(fp))[0]
        folder = os.path.dirname(os.path.dirname(fp))
        pose_path = os.path.join(
            self.data_path,
            folder,
            "pose",
            "{}.npy".format(filename_wo_extn))

        robot_pose = np.vstack([np.load(pose_path).astype(np.float32), [0,0,0,1]])
        left_local_cam = np.array([-0.1, 1, 0, 1])

        left_world_cam = (robot_pose @ left_local_cam)

        left_cam_pose = robot_pose.copy()

        left_cam_pose[:, 3] = left_world_cam

        if do_flip:
            pass #todo implement flip for pose

        return left_cam_pose.astype(np.float32)

    def check_depth(self):
        return False
        # raise NotImplementedError

    def get_depth(self, fp, do_flip):
        depth = np.array(Image.open(fp)).astype(np.float32) / 1000.0
        if do_flip:
            depth = cv2.flip(depth, 1)
        return depth

    def _get_intrinsics(self):
        # 640, 480
        w, h = self.full_res_shape

        intrinsics = np.array([[640 / w, 0., 320 / w, 0.],
                               [0., 640 / h, 240 / h, 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]], dtype="float32")
        return intrinsics




# if __name__ == '__main__':
#     data_path = '/mnt/storage2/data/jaidev/TestHabitat2'
#     segment_path = '/mnt/storage2/data/jaidev/superpixels'
#     train_filenames = ['2t7WUuJeko7/0/left_rgb/4.jpg 2t7WUuJeko7/0/left_rgb/0.jpg 2t7WUuJeko7/0/left_rgb/1.jpg 2t7WUuJeko7/0/left_rgb/2.jpg 2t7WUuJeko7/0/left_rgb/3.jpg 2t7WUuJeko7/0/left_rgb/5.jpg 2t7WUuJeko7/0/left_rgb/6.jpg 2t7WUuJeko7/0/left_rgb/7.jpg 2t7WUuJeko7/0/left_rgb/8.jpg',
#                        '2t7WUuJeko7/0/left_rgb/5.jpg 2t7WUuJeko7/0/left_rgb/1.jpg 2t7WUuJeko7/0/left_rgb/2.jpg 2t7WUuJeko7/0/left_rgb/3.jpg 2t7WUuJeko7/0/left_rgb/4.jpg 2t7WUuJeko7/0/left_rgb/6.jpg 2t7WUuJeko7/0/left_rgb/7.jpg 2t7WUuJeko7/0/left_rgb/8.jpg 2t7WUuJeko7/0/left_rgb/9.jpg']
#     height, width = 288, 384
#     frame_ids = [0, -4, -3, -2, -1, 1, 2, 3, 4]
#     num_scales = 4
#     train_dataset = HabitatTrainDataset(data_path, train_filenames, height, width, frame_ids, num_scales,
#                                         is_train=False, segment_path=segment_path, return_segment=True)
#     data = train_dataset.__getitem__(0)
#     print()
