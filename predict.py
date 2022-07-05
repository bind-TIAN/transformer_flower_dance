import os
import json
import glob
import numpy as np
from icecream import ic
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    list_type = []  # 存放每种花的种类的列表

    num_classes = 5
    im_height = im_width = 224

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)  # class_indict中存放的是类别标签

    # create model
    model = create_model(num_classes=num_classes,
                         has_logits=False)  # 这里关于has_logits的设置必须跟训练代码中的一致：即要为True则两者都为True，要为False则两者都为False
    model.build([1, 224, 224, 3])

    weights_path = './save_weights/model_tbd.ckpt'
    assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path)  # 仅仅读取权重

    # loop traverse
    for i in range(632, 1132):  # 632-1132(0-631)
        # img_path = "./test_picture/daisy/" + str(i) + ".jpg"
        # img_path = "./test_picture/roses/" + str(i) + ".jpg"
        # img_path = "./test_picture/tulips/" + str(i) + ".jpg"
        # img_path = "./test_picture/dandelion/" + str(i) + ".jpg"
        # img_path = "./test_picture/person/right" + str(i) + ".jpg"
        img_path = "./test_people/predict" + str(i) + ".jpg"
        # img_path = "./test_data3/predict" + str(i) + ".jpg"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)  # 如果路径下找不到文件则抛出异常
        img = Image.open(img_path)  # 读取一张图片
        # resize image
        img = img.resize((im_width, im_height))
        plt.imshow(img)  # 将数据显示为图像

        # read image
        img = np.array(img).astype(np.float32)  # 转换numpy数组的数据类型，转换成float32形式

        # preprocess
        img = (img / 255. - 0.5) / 0.5  # 将图片转成[-1，1]区间之间

        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img, 0))  # 表示在0位置添加数据；后面的可选数据的取值区间是[0-3]
        result = np.squeeze(model.predict(img, batch_size=1))  # 删除数组维度中的单维度条目（即维度等于1的条目），即把shape中为1的维度去掉；但是对非单维的维度不起作用
        result = tf.keras.layers.Softmax()(result)  # 使用softmax进行归一化
        predict_class = np.argmax(result)  # 返回最大值的索引
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)], result[predict_class])
        list_type.append(class_indict[str(predict_class)])
        plt.title(print_res)
        for i in range(len(result)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], result[i].numpy()))
        plt.show()
    ic(list_type)


if __name__ == '__main__':
    main()
