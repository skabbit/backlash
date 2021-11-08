import os
import sys

import numpy as np

# Root directory of the project
from PIL import Image

ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import skimage.io

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# web server with tasks to process
WEB_SERVER = "http://backlash.graycake.com"

import samples.coco.coco as coco
import samples.police2.backlash as backlash


class MaskRCNNModel():
    MODEL_FILE_NAME = "mask_rcnn_coco.h5"
    CLASSES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

    CONFIG_CLASS = coco.CocoConfig

    def __init__(self, model_file_name=None):
        # Local path to trained weights file
        COCO_MODEL_PATH = model_file_name if model_file_name else os.path.join(ROOT_DIR, self.MODEL_FILE_NAME) 

        class InferenceConfig(self.CONFIG_CLASS):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = len(self.CLASSES)  # COCO has 80 classes

        self.config = InferenceConfig()
        #         config.display()

        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)

        # Load weights
        model.load_weights(COCO_MODEL_PATH, by_name=True)

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        class_names = self.CLASSES

        self.model = model


class BacklashMaskRCNNModel(MaskRCNNModel):
    CLASSES = ['BG', 'policeman']
    MODEL_FILE_NAME = "logs/police220211106T1204/mask_rcnn_police2_0019.h5"
    CONFIG_CLASS = backlash.PoliceConfig


# IMAGE_DIR = os.path.join(ROOT_DIR, "datasets/police/val")
# file_names = next(os.walk(IMAGE_DIR))[2]
# file_names = list(filter(lambda x: not x.endswith("json"), file_names))
# image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[0]))


def process_image(image, color=(1.0, 1.0, 0.0)):
    # Run detection
    global model_full, model

    if model_full is None:
        model = BacklashMaskRCNNModel()
        model_full = MaskRCNNModel()

    results = model_full.model.detect([image], verbose=1)
    results_policeman = model.model.detect([image], verbose=1)

    mask_other = np.logical_or.reduce(results[0]['masks'][:,:,results[0]['class_ids'] == 1], axis=2)
    mask_policeman = np.logical_or.reduce(results_policeman[0]['masks'][:,:,results_policeman[0]['scores'] > 0.5], axis=2)

    # mask = np.logical_or(mask_policeman, mask_other)
    # plt.imshow(mask.astype(np.uint8))
    # masked_image = image.astype(np.uint32).copy()
    # masked_image = visualize.apply_mask(masked_image, mask, visualize.random_colors(2)[0], alpha=1)
    # # plt.imshow(masked_image.astype(np.uint8))

    mask = np.logical_and(mask_other, np.logical_not(mask_policeman))
    masked_image = image.astype(np.uint32).copy()
    masked_image = visualize.apply_mask(masked_image, mask, color, alpha=1)

    # plt.imshow(masked_image.astype(np.uint8))

    return masked_image

if __name__ == '__main__':
    image = process_image(skimage.io.imread("images/no-cops-test.jpg"))
    image = Image.fromarray(image.astype(np.uint8))
    image.save("test.jpg")