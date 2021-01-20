# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

import networks
from layers import disp_to_depth


def parse_args():
    parser = argparse.ArgumentParser(description='Inference on images in a directory.')
    parser.add_argument('--output_dir', type=str,
                        default='./outputs',
                        help='path to output directory')
    parser.add_argument('--data_dir', type=str, default='',
                        help='path to root directory of images')
    parser.add_argument('--load_weights_folder', type=str,
                        default='ckpts/weights_5f',
                        help='path of the pretrained model to use')
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument('--eval_split_path', type=str, default='',
                        help='path to txt file containing filepaths for evaluation')

    return parser.parse_args()


def prepare_model_for_test(args, device):
    model_path = args.load_weights_folder
    print("-> Loading model from ", model_path)
    # model_path = os.path.join("ckpts", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    decoder_path = os.path.join(model_path, "depth.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    decoder_dict = torch.load(decoder_path, map_location=device)

    encoder = networks.ResnetEncoder(18, False)
    decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, 
        scales=range(1),
    )

    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    decoder.load_state_dict(decoder_dict)
    
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    
    return encoder, decoder, encoder_dict['height'], encoder_dict['width']


def inference(args):
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    encoder, decoder, thisH, thisW = prepare_model_for_test(args, device)

    with open(args.eval_split_path, 'r') as f:
        filepaths = f.readlines()

    with torch.no_grad():
        for filepath in filepaths:

            folder, fileidx = filepath.split()

            output_dir = args.output_dir
            os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

            # Load image and preprocess
            input_image = pil.open(os.path.join(args.data_dir, 'imgs', folder, '{}.png'.format(fileidx))).convert('RGB')
            original_width, original_height = np.load(os.path.join(args.data_dir, 'depths', folder,
                                                                   '{}.npy'.format(fileidx))).shape[::-1]
            input_image = input_image.resize((thisH, thisW), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            outputs = decoder(encoder(input_image))

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            name_dest_npy = os.path.join(output_dir, folder, '{}.npy'.format(fileidx))
            print("-> Saving depth npy to ", name_dest_npy)
            _, scaled_depth = disp_to_depth(disp_resized, 0.1, 10)
            np.save(name_dest_npy, scaled_depth.cpu().numpy()[0, 0, :, :])

            # Saving colormapped depth image
            depth = scaled_depth.squeeze().cpu().numpy()
            vmax = np.percentile(depth, 95)
            normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_dir, folder, '{}.png'.format(fileidx))
            print("-> Saving depth png to ", name_dest_im)
            im.save(name_dest_im)

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    inference(args)
