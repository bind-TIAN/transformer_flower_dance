import splitfolders

if __name__ == "__main__":
    base_dir = 'D:/deep-learning-for-image-processing-master/tensorflow_classification/vision_transformer/'
    splitfolders.ratio(input=base_dir + 'data_set/flower_data/0Z6Jna', output='outputs', seed=2022,
                       ratio=(0.8, 0.2))
    # 以上代码将花数据集划分成train和val两个集合

