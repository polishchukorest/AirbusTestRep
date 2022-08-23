'''
   Orest Polishchuk (orest.polishchuk98@gmail.com)
   inference script for the model usage, segmentation of the single image

   sources: https://www.youtube.com/watch?v=T6h-mVVpafI&list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG&index=121&ab_channel=Apeer_micro;
            https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial117_building_unet_using_encoder_decoder_blocks.ipynb;
            https://www.kaggle.com/code/iafoss/unet34r-ship-detection
            https://keras.io/examples/vision/oxford_pets_image_segmentation/
'''

#imports
import keras.models
from keras.models import load_model
from keras.utils import load_img
from train import build_unet
from utils import get_f1
import matplotlib.pyplot as plt
import numpy as np

#checking every probability > 0.5 and returning an array of 0's and 1's
def prediction_convert(pred, size, threshold):
    pred_image = np.resize(pred, new_shape=(size))
    pred_image = np.array((pred_image > threshold), dtype=float)
    return pred_image


if __name__ == "__main__":
    IMG_SIZE = 256

    # downloading pretrained model
    unet = keras.models.load_model("model_saved", custom_objects={'get_f1': get_f1})

    # loading an example image from the test directory
    img = np.array(load_img('test_v2//00a3ab3cc.jpg', target_size=(256, 256)))
    img = np.expand_dims(img, axis=0)

    # getting prediction probabilities for 1 image (as __call__ method is better for that purpose)
    pred = unet.__call__(img)

    # showing a segmented image with threshold of probability 0.5
    true_pred = prediction_convert(pred, size = (IMG_SIZE, IMG_SIZE, 1), threshold=0.5)
    imgplot = plt.imshow(true_pred)
    plt.show()
