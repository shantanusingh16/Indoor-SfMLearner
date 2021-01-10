import os
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic, watershed, felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

# graph
from skimage.future import graph
from skimage.color import gray2rgb
import networkx as nx


from concurrent.futures import ProcessPoolExecutor
import tqdm
from functools import partial

import matplotlib.pyplot as plt

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int,
                    help='number of cpu workers',
                    default=6)
parser.add_argument('--data_dir', type=str,
                    help='path to scannet data',
                    required=True)
parser.add_argument('--output_dir', type=str,
                    help='where to store extracted segment',
                    required=True)
args = parser.parse_args()

output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def image2seg(folder, filename):
    image = cv2.imread(os.path.join(tgt_dir, folder, filename))
    image = cv2.resize(image, (640, 480))
    segment = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)

    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    filename_wo_extn = os.path.splitext(os.path.basename(filename))[0]
 
    np.savez(os.path.join(output_dir, folder, "seg_{}.npz".format(filename_wo_extn)), segment_0 = segment)
    return


# multi processing fitting
executor = ProcessPoolExecutor(max_workers=args.num_workers)
futures = []

tgt_dir = os.path.join(args.data_dir, 'imgs')
all_files = [(folder, filename) for folder in os.listdir(tgt_dir) for filename in
             os.listdir(os.path.join(tgt_dir, folder))]

for folder, filename in all_files:
    task = partial(image2seg, folder, filename)
    futures.append(executor.submit(task))

results = []
[results.append(future.result()) for future in tqdm.tqdm(futures)]
