import os
from icecream import ic
import cv2
import numpy as np

if __name__ == "__main__":
    # dir = "D:/deep-learning-for-image-processing-master/pytorch_segmentation/deeplab_v3/boat_dataset/"#train_dataset_seg/cat_dataset_seg/dog_dataset_seg
    # dir = "D:/deep-learning-for-image-processing-master/tensorflow_classification/vision_transformer/boat_dataset_seg/"
    # dir = "D:/deep-learning-for-image-processing-master/pytorch_segmentation/deeplab_v3/train_dataset/"
    dir = "D:/deep-learning-for-image-processing-master/tensorflow_classification/vision_transformer/data_set/flower_data/tianbingdi/people/"
    filelist = os.listdir(dir)
    for file in filelist:
        img = cv2.imread(dir + file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros_like(img)
        img2[:, :, 0] = gray
        img2[:, :, 1] = gray
        img2[:, :, 2] = gray
        cv2.imwrite(dir + file, img2)
