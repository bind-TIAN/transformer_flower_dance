import os

from icecream import ic
from PIL import Image

if __name__ == "__main__":
    # path = "./test_picture/person/"
    # path = "./data_set/flower_data/object/cat/"
    path = "D:/deep-learning-for-image-processing-master/tensorflow_classification/vision_transformer/test_data3/"
    filelist = os.listdir(path)
    for file in filelist:
        img = Image.open(path + file)
        try:
            r, g, b = img.split()
        except Exception:
            ic(file)  # 显示不是三通道图片的图片名
