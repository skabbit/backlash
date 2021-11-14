import io
import cv2
import numpy as np
from PIL import Image
import skvideo.io
from tqdm import tqdm
import argparse

from utils import Backlash2MaskRCNNModel
from mrcnn import visualize

# a little bit smoother video processing
# reducing shuttering effect by remembering previous masks

parser = argparse.ArgumentParser()
parser.add_argument("--input", dest='input', type=str, default="test.mp4")
parser.add_argument("--output", dest='output', type=str, default="output.mp4")
parser.add_argument("--model", dest='model', type=str, default=None)
parser.add_argument("--limit", dest='limit', type=int, default=None)
parser.add_argument("--skip", dest='skip', type=int, default=None)
parser.add_argument("--color", dest='color', type=lambda x: tuple(map(float, x.split(','))), default=(0., 1., 0.))
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

model = Backlash2MaskRCNNModel(args.model)


mask_protester_last = None

for frame in tqdm(videogen, total=maximum):
    current += 1
    if current > maximum:
        break
    image = np.asarray(Image.fromarray(frame).convert('RGB'))

    results = model.model.detect([image], verbose=0)

    mask_protester = np.logical_or.reduce(results[0]['masks'][:,:,results[0]['class_ids'] == 2], axis=2)
    mask_protester_1 = mask_protester

    if mask_protester_last is None:
        mask_protester_last = mask_protester

    mask_protester = np.logical_or(mask_protester, mask_protester_last)

    masked_image = image.astype(np.uint32).copy()
    masked_image = visualize.apply_mask(masked_image, mask_protester, args.color, alpha=1)

    writer.writeFrame(masked_image)

    mask_protester_last = mask_protester_1

writer.close()

