'''
   Orest Polishchuk (orest.polishchuk98@gmail.com)
   train script for semantic segmentation using U-net architecture

   sources: https://www.youtube.com/watch?v=T6h-mVVpafI&list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG&index=121&ab_channel=Apeer_micro;
            https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial117_building_unet_using_encoder_decoder_blocks.ipynb;
            https://www.kaggle.com/code/iafoss/unet34r-ship-detection
'''

import os
import pandas as pd
from utils import *
from dataset import *
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adamax
from keras.layers import Activation, MaxPool2D, Concatenate
from keras.losses import BinaryFocalCrossentropy
from keras.metrics import BinaryIoU
import tensorflow as tf
import tensorflow_addons as tfa

from sklearn.model_selection import train_test_split


# convolution block for U-net architecture
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


# encoder block from U-net
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


# decoder block
def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


# main function for implementing the U-net
def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)  # Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    if n_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model


if __name__ == "__main__":
    # file paths, number constants
    TRAIN = 'train_v2'
    TEST = 'test_v2'
    SEGMENTATION = 'train_ship_segmentations_v2.csv'
    IMG_SIZE = 256
    TRAIN_SET_SIZE = 7999
    NUM_WORKERS = 2

    # train and validation set
    input_img_paths = sorted(
        [
            os.path.join(TRAIN, fname)
            for fname in os.listdir(TRAIN)
            if fname.endswith(".jpg")
        ]
    )[:TRAIN_SET_SIZE]  # :size of the training set

    # split for train and val set, creation of dataframe with targets
    train, val = train_test_split(input_img_paths, test_size=0.05, random_state=0)
    segmentation_df = pd.read_csv(SEGMENTATION).set_index('ImageId')

    # removing images without any ships
    train = cut_empty(train, segmentation_df)
    val = cut_empty(val, segmentation_df)

    # callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint("airbus_detection.h5", save_best_only=True)
    ]

    # instantiation of train and val AirbusDatasets
    train = AirbusDataset(8, IMG_SIZE, train, segmentation_df)
    val = AirbusDataset(8, IMG_SIZE, val, segmentation_df)

    # building unet
    unet = build_unet((IMG_SIZE, IMG_SIZE, 3), 1)

    # training phase
    unet.compile(optimizer=Adamax(learning_rate=1e-4), loss=BinaryFocalCrossentropy(),
                 metrics=[get_f1, BinaryIoU()])
    # unet.compile(optimizer=Adamax(learning_rate=1e-4), loss=BinaryFocalCrossentropy(),
    #             metrics=[tfa.metrics.F1Score(num_classes=1), BinaryIoU()])
    unet.fit(train, validation_data=val, epochs=3, callbacks=callbacks, workers=NUM_WORKERS)
    unet.save('model_saved_final')
