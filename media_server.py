import os
import time
import io
import random

import numpy as np
import requests
from PIL import Image

from mrcnn import visualize
from utils import process_image

from local_config import WEB_SERVER, PROXY


done = False
while not done:
    time.sleep(1)
    try:
        job = requests.get(WEB_SERVER + "/jobs", proxies=PROXY)
    except requests.exceptions.ConnectionError:
        continue

    if job.text == "empty":
        continue

    filename = job.text
    sha256 = os.path.splitext(filename)[0]

    response = requests.get(WEB_SERVER + "/static/uploads/" + filename)
    image = Image.open(io.BytesIO(response.content)).convert('RGB')

    result_array = process_image(np.asarray(image), visualize.random_colors(random.randint(3, 20))[2])
    result_image = Image.fromarray(result_array.astype(np.uint8))
    stream = io.BytesIO()
    stream.name = sha256 + ".jpeg"
    result_image.save(stream)
    requests.post(WEB_SERVER + "/jobs", files={'file': (sha256 + ".jpeg", stream.getbuffer())})

