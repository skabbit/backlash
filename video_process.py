import io
import cv2
import numpy as np
from PIL import Image
import skvideo.io
from tqdm import tqdm
import argparse

from utils import process_image

parser = argparse.ArgumentParser()
parser.add_argument("--input", dest='input', type=str, default="test.mp4")
parser.add_argument("--output", dest='output', type=str, default="output.mp4")
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

for frame in tqdm(videogen, total=maximum):
    image = Image.fromarray(frame).convert('RGB')
    result_array = process_image(np.asarray(image))
    writer.writeFrame(result_array)
    current += 1
    if current == maximum:
        break

writer.close()

