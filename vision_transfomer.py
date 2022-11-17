# vision transformer model
# Path: vision_transfomer.py

import tensorflow as tf
import tensorflow.python.keras.layers as layers
# import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import os
import time
import datetime
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D
#import layernormalization
from tensorflow.keras.layers import LayerNormalization


# check cuda
print(tf.test.is_gpu_available())

# load image

img = tf.keras.preprocessing.image.load_img('C:\study\Mask-RCNN-TF2-master\images/2383514521_1fc8d7b0de_z.jpg')

#640x428 -> 640x640

img_resized = tf.image.resize(img, [640, 640])

# patching 16x16
patch_size = 16
num_patches = (640 // patch_size) ** 2
print(num_patches)

# patching
patches = tf.image.extract_patches(
    images=[img_resized],
    sizes=[1, patch_size, patch_size, 1],
    strides=[1, patch_size, patch_size, 1],
    rates=[1, 1, 1, 1],
    padding='VALID',
)

print(patches.shape)

reshape_patches = tf.reshape(patches, [num_patches, patch_size * patch_size * 3])

print(reshape_patches.shape)

# cls token
cls_token = tf.Variable(tf.random.normal([1, 1, 768]))
print(cls_token.shape)

# concat cls token
reshape_cls_token = tf.reshape(cls_token, [1, 768])
concat_cls_token = tf.concat([reshape_cls_token, reshape_patches], axis=0)
print(concat_cls_token.shape)

# positional embedding
positional_embedding = tf.Variable(tf.random.normal([num_patches + 1, 768]))
print(positional_embedding.shape)

# add positional embedding
add_positional_embedding = concat_cls_token + positional_embedding
print(add_positional_embedding.shape)

# transformer
# embedding layer
embedding_layer = tf.keras.layers.Dense(768)
embedding = embedding_layer(add_positional_embedding)
print(embedding.shape)

# multi head attention
multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=12, key_dim=64)

# feed forward network
feed_forward_network = tf.keras.Sequential([Dense(3072, activation="relu"), Dense(768)])

# layer normalization
layer_normalization = LayerNormalization()

# dropout
dropout = Dropout(0.1)

# pretrained model
# Path: vision_transfomer.py

# import pretrained model ImageNet-21k

imgnet21k = tf.keras.applications.ViTModel(





# # transformer block
# class VisionTransformer(tf.keras.Model):
#     def __init__(self, num_layers, embedding_dim, num_heads, mlp_dim, image_size, patch_size, num_classes):
#         super(VisionTransformer, self).__init__()
#         self.num_layers = num_layers
#         self.patch_size = patch_size
#         self.num_patches = (image_size // patch_size) ** 2
#         self.cls_token = tf.Variable(tf.random.normal([1, 1, embedding_dim]))
#         self.positional_embedding = tf.Variable(tf.random.normal([self.num_patches + 1, embedding_dim]))
#         self.embedding_layer = tf.keras.layers.Dense(embedding_dim)
#         self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=64)
#         self.feed_forward_network = tf.keras.Sequential([Dense(mlp_dim, activation="relu"), Dense(embedding_dim)])
#         self.layer_normalization = LayerNormalization()
#         self.dropout = Dropout(0.1)
#         self.mlp_head = tf.keras.Sequential([Dense(mlp_dim, activation="relu"), Dense(num_classes)])

#     def call(self, images):
#         # patching
#         patches = tf.image.extract_patches(
#             images=images,
#             sizes=[1, self.patch_size, self.patch_size, 1],
#             strides=[1, self.patch_size, self.patch_size, 1],
#             rates=[1, 1, 1, 1],
#             padding='VALID',
#         )
#         reshape_patches = tf.reshape(patches, [self.num_patches, self.patch_size * self.patch_size * 3])
#         # concat cls token
#         reshape_cls_token = tf.reshape(self.cls_token, [1, 768])
#         concat_cls_token = tf.concat([reshape_cls_token, reshape_patches], axis=0)
#         # add positional embedding
#         add_positional_embedding = concat_cls_token + self.positional_embedding
#         # embedding
#         embedding = self.embedding_layer(add_positional_embedding)
#         # transformer
#         for _ in range(self.num_layers):
#             x1 = self.multi_head_attention(embedding, embedding)
#             x2 = self.layer_normalization(embedding + x1)
#             x3 = self.feed_forward_network(x2)
#             x4 = self.layer_normalization(x2 + x3)
#             embedding = self.dropout(x4)
#         # mlp head
#         x = self.mlp_head(embedding[:, 0])
#         return x
    
# # model
# model = VisionTransformer(num_layers=12, embedding_dim=768, num_heads=12, mlp_dim=3072, image_size=640, patch_size=16, num_classes=1000)

# # compile
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # summary
# model.summary()


