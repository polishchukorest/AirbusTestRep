# AirbusTestRep
My solution  

Introduction:  
To test my solution two directories with original images(test_v2, train_v2) should be copied to the project as well as train_ship_segmentations_v2.csv  
All this can be downloaded from https://www.kaggle.com/c/airbus-ship-detection/data  
My solution did not use any GPU resources because tensorflow couldn't detect CUDA library dlls. So, there was only 3 epochs with approximately 1-2k images, batch was 8 images.  
  
1. Exploratory data analysis:  
   1.1. I've checked few rows of data and loaded a random image;  
   1.2. Most of analysis were taken from already existiting notebook: [1]  
2. Python project: creation of AirbusDataset class to iterate over given images:  
   2.1. AirbusData set was created (this keras guide [2] was taken as a reference).  
   Few details were changed, as data was coming as img and dataframe row with encoded pixels. Reshaping was also used, as training on original images would take too much time.  
 
3. Implementation of build_unet function:  
   3.1. implemetation of UNET architecture was taken from here: [3]  
   This implementation is using functions such as conv_block, encoder_block, decoder_block to build typical parts of U-net architecture.   
4. Implementation of other useful utilities:  
   4.1. get_mask function to return 1bit image from the given dataframe. Source: [1]  
   4.2. get_f1 function to implement dice(f1) score. Source: [4]  
   4.3. cut_empty to return list of images that contain ships. Also was taken from [1]  
5. Training:  
   5.1. Useful constants were created for training dataset size, num workers, cropped imgage size, etc.  
   5.2. Only images with ships in them were taken.  
   5.3 DataSet object was instantiated, model was compiled, trained and saved, metrics was(dice and IoU).  
     
   Training logs:  
   
   Epoch 1/3
207/207 [==============================] - 3694s 18s/step - loss: 0.0440 - get_f1: 0.0739 - binary_io_u: 0.4888 - val_loss: 0.0759 - val_get_f1: 0.0367 - val_binary_io_u: 0.4817
Epoch 2/3
207/207 [==============================] - 3738s 18s/step - loss: 0.0113 - get_f1: 0.1274 - binary_io_u: 0.5360 - val_loss: 0.0324 - val_get_f1: 0.0490 - val_binary_io_u: 0.4964
Epoch 3/3
207/207 [==============================] - 3731s 18s/step - loss: 0.0083 - get_f1: 0.2416 - binary_io_u: 0.5732 - val_loss: 0.0122 - val_get_f1: 0.1693 - val_binary_io_u: 0.5332

6 Inference:
   6.1. prediction_convert is taking an array of predictions and returning 1 bit image, with def. threshold 0.5.
   6.2. I loaded the model, 1 random image from test was taken, the model outputed a prediction resulting in a corresponding 1bit mask.
   

Sources:
https://www.kaggle.com/code/iafoss/unet34r-ship-detection [1]
https://keras.io/examples/vision/oxford_pets_image_segmentation/ [2]
https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial117_building_unet_using_encoder_decoder_blocks.ipynb [3]
https://aakashgoel12.medium.com/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d [4]
