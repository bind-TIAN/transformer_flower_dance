代码说明：
1. channel_transfer_one_to_three.py: 将一通道图片转成三通道
2. check_channels_img.py: 检查图片是否为三通道
3. open_h5.py: 打开后缀为.h5类型的文件
4. split-data.py: 批量将图片文件夹划分为训练集,测试集和验证集
5. predict_copy.py: 预测代码的copy文件，方便恢复出厂设置

测试文件说明：
1.  daisy, danlelion, roses, sunflowers, tulips分别为：雏菊，蒲公英，玫瑰，向日葵，郁金香这五种花卉的数据集。
    person为行人识别测试数据集，其中是采用语义分割之后的50张行人图片。
    访问路径为：./test_picture/xx

2.  flower_photos为五种花卉的训练数据集文件夹
    object为[船，猫，狗，人，火车]五种训练数据集的文件夹，里面是语义分割处理之后的一通道图片
    访问路径为：./data_set/flower_data/xx
