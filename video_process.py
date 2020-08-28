import io
import cv2
import numpy as np
from PIL import Image
import skvideo.io
from tqdm import tqdm
import argparse

from utils import process_image

# parser = argparse.ArgumentParser()
# parser.add_argument("--input", dest='str', default="test.mp4")
# parser.add_argument("--limit", dest='int', default=None)
# args = parser.parse_args()


probe = skvideo.io.ffprobe('Russia Riot Police Use Batons at Moscow Protest-A_Zf4EusWJo.mp4')

videogen = skvideo.io.vreader('Russia Riot Police Use Batons at Moscow Protest-A_Zf4EusWJo.mp4')
writer = skvideo.io.FFmpegWriter("outputvideo.mp4", outputdict={
    '-vcodec': 'libx264', '-b': '5000000'
})

total = int(probe['video']['@nb_frames'])
maximum = 3
current = 0
for frame in tqdm(videogen, total=maximum):
    image = Image.fromarray(frame).convert('RGB')
    result_array = process_image(np.asarray(image))
    writer.writeFrame(result_array)
    current += 1
    if current == maximum:
        break

writer.close()

