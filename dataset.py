'''
   Orest Polishchuk (orest.polishchuk98@gmail.com)
   train script for semantic segmentation using U-net architecture

   sources: https://www.youtube.com/watch?v=T6h-mVVpafI&list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG&index=121&ab_channel=Apeer_micro;
            https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial117_building_unet_using_encoder_decoder_blocks.ipynb;
            https://www.kaggle.com/code/iafoss/unet34r-ship-detection
            https://keras.io/examples/vision/oxford_pets_image_segmentation/
'''

import os
from PIL import Image
from tensorflow import keras
import numpy as np
from keras.utils import load_img
import cv2
from utils import *


class AirbusDataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_df):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_df = target_df

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]

        x = np.zeros((self.batch_size,) + (self.img_size, self.img_size) + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + (self.img_size, self.img_size), dtype="int8")

        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=(self.img_size, self.img_size))
            # img = cv2.resize(np.array(img), (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            # print(img.shape)
            x[j] = img
            img_target = Image.fromarray(get_mask(path[9:], self.target_df)).resize((self.img_size, self.img_size),
                                                                                    cv2.INTER_AREA).convert('L')

            y[j] = np.array(img_target).astype(np.float32)

            # print(f"x.shape, y.shape: {img.shape, y[j].shape}")

            # y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
            # for j, path in enumerate(self.target_df):
            #    img = load_img(path, target_size=self.img_size, color_mode="grayscale")

            # y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1
        return x, y
