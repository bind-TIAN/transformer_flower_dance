"""
refer to:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import tensorflow as tf
from tensorflow import keras
from keras import Model, layers, initializers
# from tensorflow.keras import Model, layers, initializers
import numpy as np


class PatchEmbed(layers.Layer):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.img_size = (img_size, img_size)  # image_size(224*224)
        self.grid_size = (img_size // patch_size, img_size // patch_size)  # 整数除法，向下取整
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # (14*14=196)

        self.proj = layers.Conv2D(filters=embed_dim, kernel_size=patch_size,
                                  strides=patch_size, padding='SAME',
                                  kernel_initializer=initializers.LecunNormal(),
                                  bias_initializer=initializers.Zeros())  # 这里其实是一个膨胀卷积

    # 1、call函数里的inputs是啥
    def call(self, inputs, **kwargs):  # **kwargs是keywords arguments的缩写，对将来该函数中要传入多少参数不确定，即可使用可变参数
        B, H, W, C = inputs.shape  # 依次表示[批量，高度，宽度，通道数]
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."  # 判断图片必须是224*224的
        x = self.proj(inputs)  # 这一步得到一个经过卷积操作之后的新的[B,H,W,C]
        # [B, H, W, C] -> [B, H*W, C]
        x = tf.reshape(x, [B, self.num_patches, self.embed_dim])  # 进行展平操作
        return x


class ConcatClassTokenAddPosEmbed(layers.Layer):
    def __init__(self, embed_dim=768, num_patches=196, name=None):
        super(ConcatClassTokenAddPosEmbed, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.num_patches = num_patches

    # 2、build函数里面的形参：input_shape是啥
    def build(self, input_shape):
        self.cls_token = self.add_weight(name="cls",
                                         shape=[1, 1, self.embed_dim],
                                         initializer=initializers.Zeros(),
                                         trainable=True,
                                         dtype=tf.float32)  # 获取1*768形状的token
        self.pos_embed = self.add_weight(name="pos_embed",
                                         shape=[1, self.num_patches + 1, self.embed_dim],
                                         initializer=initializers.RandomNormal(stddev=0.02),
                                         trainable=True,
                                         dtype=tf.float32)  # 获取197*768形状的位置编码

    # 3、call函数里面的inputs是从哪里传进来的
    def call(self, inputs, **kwargs):
        batch_size, _, _ = inputs.shape

        # [1, 1, 768] -> [B, 1, 768]
        cls_token = tf.broadcast_to(self.cls_token,
                                    shape=[batch_size, 1, self.embed_dim])  # 将张量进行广播已得到输出张量，将token复制多份一一拼接
        x = tf.concat([cls_token, inputs], axis=1)  # [B, 197, 768]，这里使用了tf.concat函数中高维拼接的形式；注意这里的axis=1表示以纵列方式进行拼接
        x = x + self.pos_embed  # 与位置编码相加得到最终的输出

        return x


class Attention(layers.Layer):
    k_ini = initializers.GlorotUniform()  # 均匀初始化器又叫做Xavier均匀初始化器（具体介绍看CS231N视频）
    b_ini = initializers.Zeros()  # 这里的标黄警告不用管，是pycharm不能兼容的原因，代码依然可以正常被运行

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 name=None):
        super(Attention, self).__init__(name=name)
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 计算每一个header，它的dimension；计算方式：总维度➗header个数
        self.scale = qk_scale or head_dim ** -0.5  # 根号下dk的一个缩放因子，其值等于（1/根号dk）
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, name="qkv",
                                kernel_initializer=self.k_ini,
                                bias_initializer=self.b_ini)  # 实现生成wq、wk、wv，大的全连接层，一次性生成wq、wk、wv
        self.attn_drop = layers.Dropout(attn_drop_ratio)
        self.proj = layers.Dense(dim, name="out",
                                 kernel_initializer=self.k_ini,
                                 bias_initializer=self.b_ini)  # 每一个拼接之后的head通过wo进行融合得到最终的输出
        self.proj_drop = layers.Dropout(proj_drop_ratio)

    def call(self, inputs, training=None):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = inputs.shape  # 这里已经是展平之后的了，因此可以直接得到B\N\C

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        qkv = self.qkv(inputs)  # 全连接层节点的个数是我们输入的节点个数的三倍
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads,
                               C // self.num_heads])  # 由于之前是通过一个全连接层得到的qkv，因此这里要分成三份分别得到q、k和v；然后有需要针对每个q、k、v将它分成num_heads份，这是因为我们有多个head，因此需要做进一步的划分
        # transpose: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # 矩阵的不同行和不同列的转置操作，是为了后续做运算方便
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # 通过切片分别取出q、k、v，那么q、k、v它们各自的shape就是[batch_size,num_heads,num_patches+1,embed_dim_per_head]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = tf.matmul(a=q, b=k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)  # 针对attn的矩阵中的每一行进行了softmax的处理，因为这里的axis=-1
        attn = self.attn_drop(attn, training=training)

        # multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        x = tf.matmul(attn, v)
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        x = tf.transpose(x, [0, 2, 1, 3])
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = tf.reshape(x, [B, N, C])  # 将每一个head的输出在相同位置上进行一个拼接，那么就变成了total_embed_dim

        x = self.proj(x)  # 然后将我们得到的输出通过wo做进一步的融合，得到我们的输出，通过全连接层进行映射
        x = self.proj_drop(x, training=training)  # 通过dropout层得到最终的输出
        return x


class MLP(layers.Layer):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    k_ini = initializers.GlorotUniform()
    b_ini = initializers.RandomNormal(stddev=1e-6)

    def __init__(self, in_features, mlp_ratio=4.0, drop=0., name=None):
        super(MLP, self).__init__(name=name)
        self.fc1 = layers.Dense(int(in_features * mlp_ratio), name="Dense_0",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.act = layers.Activation("gelu")
        self.fc2 = layers.Dense(in_features, name="Dense_1",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.drop = layers.Dropout(drop)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x


class Block(layers.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 name=None):
        super(Block, self).__init__(name=name)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_0")  # Layer Norm
        self.attn = Attention(dim, num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio,
                              name="MultiHeadAttention")  # Multi-Head Attention
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = layers.Dropout(rate=drop_path_ratio, noise_shape=(None, 1, 1)) if drop_path_ratio > 0. \
            else layers.Activation(
            "linear")  # DropPath，其中的noise_shape=(batch维度，num_patches+1，embed_dimension)，该方法是一个随机深度的dropout方法
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_1")
        self.mlp = MLP(dim, drop=drop_ratio, name="MlpBlock")

    def call(self, inputs, training=None):
        x = inputs + self.drop_path(self.attn(self.norm1(inputs)), training=training)
        x = x + self.drop_path(self.mlp(self.norm2(x)), training=training)
        return x


class VisionTransformer(Model):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768,
                 depth=12, num_heads=12, qkv_bias=True, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 representation_size=None, num_classes=1000,
                 name="ViT-B/16"):  # 这里的depth表示encoder block重复堆叠的次数；num_heads表示MLP Head模块中头的个数
        super(VisionTransformer, self).__init__(name=name)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.qkv_bias = qkv_bias

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      embed_dim=embed_dim)  # 调用PatchEmbed函数实例化patches
        num_patches = self.patch_embed.num_patches  # patches的数量
        self.cls_token_pos_embed = ConcatClassTokenAddPosEmbed(embed_dim=embed_dim,
                                                               num_patches=num_patches,
                                                               name="cls_pos")  # 实例化concat，拼接并且加上position embedding，需要两个参数：embed_dim（768）、num_patches（196）

        self.pos_drop = layers.Dropout(drop_ratio)  # 随机将某些神经元设置为0防止过拟合，这里是Dropout层

        dpr = np.linspace(0., drop_path_ratio, depth)  # stochastic depth decay rule（随机深度衰减规则）构建等差序列
        self.blocks = [Block(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                             drop_path_ratio=dpr[i], name="encoderblock_{}".format(i))
                       for i in range(depth)]  # 通过for循环来构建transformer encoder；通过depth来创建堆叠多少个blocks

        self.norm = layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")  # 定义一个Layer Norm

        if representation_size:
            self.has_logits = True
            self.pre_logits = layers.Dense(representation_size, activation="tanh", name="pre_logits")  # 这是一个全连接层
        else:
            self.has_logits = False
            self.pre_logits = layers.Activation("linear")  # 否则的话就不做处理，相当于没有pre-logits这个模块

        self.head = layers.Dense(num_classes, name="head", kernel_initializer=initializers.Zeros())

    def call(self, inputs, training=None):
        # [B, H, W, C] -> [B, num_patches, embed_dim]
        x = self.patch_embed(inputs)  # [B, 196, 768]
        x = self.cls_token_pos_embed(x)  # [B, 197, 768]
        x = self.pos_drop(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)

        return x


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes,
                              name="ViT-B_16")
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes,
                              name="ViT-B_32")
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):  # ImageNet数据集
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes,
                              name="ViT-L_16")
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes,
                              name="ViT-L_32")
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes,
                              name="ViT-H_14")
    return model
