import io
import cv2
import numpy as np
from PIL import Image
import skvideo.io
from tqdm import tqdm
import argparse

from utils import BacklashMaskRCNNModel, MaskRCNNModel
from mrcnn import visualize

# a little bit smoother video processing
# reducing shuttering effect by remembering previous masks

parser = argparse.ArgumentParser()
parser.add_argument("--input", dest='input', type=str, default="test.mp4")
parser.add_argument("--output", dest='output', type=str, default="output.mp4")
parser.add_argument("--model", dest='model', type=str, default=None)
parser.add_argument("--limit", dest='limit', type=int, default=None)
args = parser.parse_args()

probe = skvideo.io.ffprobe(args.input)
videogen = skvideo.io.vreader(args.input)
writer = skvideo.io.FFmpegWriter(args.output, outputdict={
    '-vcodec': 'libx264',
    '-pix_fmt': 'yuv420p',
})

total = int(probe['video']['@nb_frames'])
maximum = args.limit if args.limit else total
current = 0
color = (0., 1., 0.)

model = BacklashMaskRCNNModel(args.model)
model_full = MaskRCNNModel()


mask_other_last = None
mask_policeman_last = None

for frame in tqdm(videogen, total=maximum):
    image = np.asarray(Image.fromarray(frame).convert('RGB'))

    results = model_full.model.detect([image], verbose=0)
    results_policeman = model.model.detect([image], verbose=0)

    mask_other = np.logical_or.reduce(results[0]['masks'][:,:,results[0]['class_ids'] == 1], axis=2)
    mask_policeman = np.logical_or.reduce(results_policeman[0]['masks'][:,:,results_policeman[0]['scores'] > 0.5], axis=2)
    mask_other_1 = mask_other
    mask_policeman_1 = mask_policeman

    if mask_other_last is None:
        mask_other_last = mask_other
    if mask_policeman_last is None:
        mask_policeman_last = mask_policeman

    mask_other = np.logical_or(mask_other, mask_other_last)
    mask_policeman = np.logical_or(mask_policeman, mask_policeman_last)

    mask = np.logical_and(mask_other, np.logical_not(mask_policeman))
    masked_image = image.astype(np.uint32).copy()
    masked_image = visualize.apply_mask(masked_image, mask, color, alpha=1)

    writer.writeFrame(masked_image)
    current += 1
    if current == maximum:
        break

    mask_other_last = mask_other_1
    mask_policeman_last = mask_policeman_1

writer.close()

