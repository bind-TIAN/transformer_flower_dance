import os
import re
import sys
import math
import datetime

import tensorflow as tf
from tqdm import tqdm

from vit_model import vit_base_patch16_224_in21k as create_model
from utils import generate_ds

assert tf.version.VERSION >= "2.4.0", "version of tf must greater/equal than 2.4.0"


def main():
    data_root = "./data_set/flower_data/0Z6Jna"  # get data root path

    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights")  # 创建一个文件夹

    batch_size = 8
    epochs = 3  # 原来是10
    num_classes = 5
    freeze_layers = True
    initial_lr = 0.001
    weight_decay = 1e-4

    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 打印日志
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))  # 指定写入事件文件的目录

    # data generator with data augmentation
    train_ds, val_ds = generate_ds(data_root, batch_size=batch_size, val_rate=0.2)  # 图像增强

    # create model
    model = create_model(num_classes=num_classes, has_logits=False)  # 创建模型，has_logits置为False表示不使用Pre-Logits
    model.build((1, 224, 224, 3))  # 因为这里使用的是subclasses模型的构建方式，因此这里需要先手动的build一下

    # 下载我提前转好的预训练权重
    # 链接: https://pan.baidu.com/s/1ro-6bebc8zroYfupn-7jVQ  密码: s9d9
    # load weights
    pre_weights_path = './tf_vit/ViT-B_16.h5'
    assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)  # 判断freeze_layers是否为True
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

    # freeze bottom layers
    if freeze_layers:  # 若freeze_layers设置为True，那么就只会训练最后的MLP head 这个权重，其它权重就会全部被冻结；同理，若想训练所有的权重那么就把这里的freeze_layers置为false
        for layer in model.layers:
            if "pre_logits" not in layer.name and "head" not in layer.name:
                layer.trainable = False
            else:
                print("training {}".format(layer.name))

    model.summary()

    # custom learning rate curve
    def scheduler(now_epoch):
        end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine，计算更新速率
        new_lr = rate * initial_lr

        # writing lr into tensorboard
        with train_writer.as_default():
            tf.summary.scalar('learning rate', data=new_lr, step=epoch)

        return new_lr

    # using keras low level api for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # 交叉熵损失函数
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)  # 随机梯度下降，可通过CS231N学习

    train_loss = tf.keras.metrics.Mean(name='train_loss')  # 计算加权平均值
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')  # 计算多分类问题的准确率

    val_loss = tf.keras.metrics.Mean(name='val_loss')  # 同上
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')  # 同上

    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            output = model(train_images, training=True)
            # cross entropy loss
            ce_loss = loss_object(train_labels, output)  # 针对类别的crose_entropy_loss

            # l2 loss
            matcher = re.compile(".*(bias|gamma|beta).*")  # 这里的正则匹配目的是：只计算所有kernel的L2损失，不去计算bias以及LayerNorm中参数的损失
            l2loss = weight_decay * tf.add_n([  # 通过tf.add_n这个方法，将所有的L2loss进行相加，然后乘以我们指定的weight_decay得到最终的损失值
                tf.nn.l2_loss(v)  # 计算每一个参数对应的L2损失
                for v in model.trainable_variables  # 遍历模型中所有可训练的参数
                if not matcher.match(v.name)
                # 在pytorch当中，通过optimizer模块中指定weight_decay，它可以帮助我们自动计算L2损失。但是在tensorflow中就需要我们手动计算。这个计算有很多种，这里只是简单的介绍其中的一种
            ])

            loss = ce_loss + l2loss  # 将L2loss和cross_entropy_loss进行相加得到最终的损失

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(ce_loss)
        train_accuracy(train_labels, output)

    @tf.function  # 实现图执行模式
    def val_step(val_images, val_labels):
        output = model(val_images, training=False)
        loss = loss_object(val_labels, output)

        val_loss(loss)
        val_accuracy(val_labels, output)

    best_val_acc = 0.  # 统计最好的验证准确率
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        train_bar = tqdm(train_ds, file=sys.stdout)
        for images, labels in train_bar:
            train_step(images, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

        # update learning rate
        optimizer.learning_rate = scheduler(epoch)

        # validate
        val_bar = tqdm(val_ds, file=sys.stdout)
        for images, labels in val_bar:
            val_step(images, labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())
        # writing training loss and acc
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), epoch)

        # writing validation loss and acc
        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), epoch)
            tf.summary.scalar("accuracy", val_accuracy.result(), epoch)

        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            save_name = "./save_weights/model_tbd2.ckpt"
            model.save_weights(save_name, save_format="tf")


if __name__ == '__main__':
    main()
