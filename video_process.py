import io
import cv2
import numpy as np
from PIL import Image
import skvideo.io
from tqdm import tqdm
import argparse

from utils import process_image

parser = argparse.ArgumentParser()
parser.add_argument("--input", dest='str', default="test.mp4")
parser.add_argument("--limit", dest='int', default=None)
args = parser.parse_args()

probe = skvideo.io.ffprobe(args.input)
videogen = skvideo.io.vreader(args.input)
writer = skvideo.io.FFmpegWriter("outputvideo.mp4", outputdict={
    '-vcodec': 'libx264', '-b': '5000000'
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

