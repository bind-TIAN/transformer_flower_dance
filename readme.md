# Flower analyzation by using transformer architecture
---
## 代码说明：
---
*    channel_transfer_one_to_three.py: 将一通道图片转成三通道。
*    check_channels_img.py: 检查图片是否为三通道。
*    open_h5.py: 打开后缀为.h5类型的文件。
*    split-data.py: 批量将图片文件夹划分为训练集,测试集和验证集。
*    predict_copy.py: 预测代码的copy文件，源代码predict.py文件的备份。

## 测试文件说明（github中并没有上传该数据集，感兴趣的读者可自行制作）
---
*    0Z6Jna 文件夹是训练数据集，object为[船，猫，狗，人，火车]。访问路径为：./0Z6Jna/xx。
*    test_dataset 为测试数据集，object为[船，猫，狗，人，火车]。是采用语义分割处理后的图片。访问路径为：./test_dataset/xx。
*    tf_vit为存放预训练权重model的文件夹。
*    predict.py:预测文件/train.py:训练文件/trans_weights.py:训练权重文件/utils.py:数据处理文件/vit_model.py:ViT模型搭建文件。
