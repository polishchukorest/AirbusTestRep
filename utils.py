'''
   Orest Polishchuk (orest.polishchuk98@gmail.com)
   train script for semantic segmentation using U-net architecture

   sources: https://www.youtube.com/watch?v=T6h-mVVpafI&list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG&index=121&ab_channel=Apeer_micro;
            https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial117_building_unet_using_encoder_decoder_blocks.ipynb;
            https://www.kaggle.com/code/iafoss/unet34r-ship-detection
            https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras
            https://github.com/keras-team/keras/issues/3611
            https://aakashgoel12.medium.com/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d
'''
import numpy as np
import tensorflow as tf
from keras import backend as K
import pandas as pd
from torch import sigmoid
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from keras.losses import Loss


# creating mask out of the train_ship_segmentations_v2.csv
def get_mask(img_id, df):
    shape = (768, 768)
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    masks = df.loc[img_id]['EncodedPixels']
    if (type(masks) == float): return img.reshape(shape)
    if (type(masks) == str): masks = [masks]
    for mask in masks:
        s = mask.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start:start + length] = 1
    return img.reshape(shape).T

#computes dice score
def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


# function to cut examples without ships
def cut_empty(names, segmentation_df):
    return [name for name in names
            if (type(segmentation_df.loc[name.split('\\')[-1]]['EncodedPixels']) != float)]


if __name__ == "__main__":
    # test to chck get_mask functiong
    SEGMENTATION = 'train_ship_segmentations_v2.csv'
    segmentation_df = pd.read_csv(SEGMENTATION, sep=',').set_index('ImageId')
    img = get_mask('e0a074340.jpg', segmentation_df)
    print(img.shape)
    imgplot = plt.imshow(img)
    plt.show()
